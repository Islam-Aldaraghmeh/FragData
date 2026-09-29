"""Optional Glacier loading and candidate prediction for Experiment 1."""
import warnings
import numpy as np

from .candidates import _unpack_bag
from src.models.assets import asset_path as _asset_path
from src.models.glacier_api import GlacierAPI


def load_glacier_model(upstream_path=None, checkpoint_path=None, *,
                       aromatic_fix=False, radical_fix=False):
    """Load and return the repo's verified, frozen CPU GlacierAPI model wrapper.

    Explicit paths override GLACIER_UPSTREAM / GLACIER_CHECKPOINT and repository-local
    defaults. The existing API checks the source commit and checkpoint
    SHA256. Loading happens here, so repeated prediction calls reuse the weights.
    aromatic_fix allows the audited aromatic/Kekule fallback; radical_fix allows
    direct +1, one-radical states. Both default False. Successful baseline states
    always stay unchanged. Inspect glacier.recovery_settings to log an ablation.
    """
    upstream = _asset_path(upstream_path, "GLACIER_UPSTREAM", "vendor/ms-pred")
    checkpoint = _asset_path(checkpoint_path, "GLACIER_CHECKPOINT", "checkpoints/best.ckpt")
    glacier = GlacierAPI(upstream, checkpoint,
                         aromatic_fix=aromatic_fix, radical_fix=radical_fix)
    _ = glacier.model
    return glacier


def predict_candidate_bags(glacier, candidate_bags, *, collision_energy,
                           instrument="Orbitrap", max_states=4,
                           on_unsupported="none"):
    """Return [(bag_mz, [Nx2 spectrum or None], ms3_spectra)] at the same indices.

    collision_energy is an explicit nonnegative scalar in the model's input
    convention, shared by these predictions. It is not inferred from MSnLib
    metadata. The adapter follows glacier.aromatic_fix and glacier.radical_fix,
    always trying baseline states first. It uses [M]+ and averages individually
    normalized state spectra (max_states=4 by default).
    Only unsupported ionic-state hypotheses may become None. Other ValueErrors
    include the bag/candidate index; other exception types propagate unchanged.
    Use on_unsupported='raise' for strict use.
    """
    from src.chemistry.fragment_adapter import FragmentGlacierAdapter

    ce = float(collision_energy)
    if not np.isfinite(ce) or ce < 0:
        raise ValueError("collision_energy must be a finite, nonnegative scalar")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if on_unsupported not in {"none", "raise"}:
        raise ValueError("on_unsupported must be 'none' or 'raise'")
    adapter = FragmentGlacierAdapter(glacier, max_states=max_states)
    predictions = []
    for bag_index, bag in enumerate(candidate_bags):
        mz, fragments, ms3_spectra = _unpack_bag(bag)
        spectra = []
        for fragment_index, fragment in enumerate(fragments):
            try:
                predicted = adapter.predict(fragment, collision_energy=ce, instrument=instrument)
            except ValueError as error:
                context = (f"bag {bag_index}, candidate {fragment_index} "
                           f"({fragment.candidate_id}, m/z {mz}): {error}")
                # This is the adapter's explicit no-representable-state failure.
                if on_unsupported == "none" and str(error).startswith("No represented states:"):
                    warnings.warn(context, RuntimeWarning, stacklevel=2)
                    spectra.append(None)
                    continue
                raise ValueError(context) from error
            spectra.append(predicted.as_array())
        predictions.append((mz, spectra, ms3_spectra))
    return predictions
