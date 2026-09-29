"""Isolated spectrum-03 mixture warning audit; never edits production code.

The shadow fit replaces only its two explicit matrix products with einsum.
SciPy NNLS remains the same solver, so this is not a separate optimizer proof.
"""
import ast
from contextlib import redirect_stdout
import hashlib
import inspect
import io
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery_lab.stress_pipeline import namespace, load_scans, write_json, OUT, budget


def captured(function):
    with warnings.catch_warnings(record=True) as caught, np.errstate(all="warn"):
        warnings.simplefilter("always")
        value = function()
    return value, [dict(message=str(w.message), category=w.category.__name__,
                        filename=w.filename, line=w.lineno) for w in caught]


def array_summary(value):
    value = np.asarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                finite=bool(np.isfinite(value).all()), minimum=float(value.min()),
                maximum=float(value.max()))


def compare(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return dict(both_finite=bool(np.isfinite(left).all() and np.isfinite(right).all()),
                exact=bool(np.array_equal(left, right)),
                max_absolute_difference=float(np.max(np.abs(left - right), initial=0)),
                allclose_1e_12=bool(np.allclose(left, right, atol=1e-12, rtol=1e-12)))


def einsum_matmul(left, right):
    if left.ndim == 2 and right.ndim == 1:
        return np.einsum("ij,j->i", left, right, optimize=False)
    if left.ndim == 2 and right.ndim == 2:
        return np.einsum("ij,jk->ik", left, right, optimize=False)
    raise ValueError("Unexpected audit product dimensions")


def shadow_fit(mixture):
    class ReplaceProducts(ast.NodeTransformer):
        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.MatMult):
                return ast.copy_location(ast.Call(func=ast.Name(id="einsum_matmul", ctx=ast.Load()),
                    args=[node.left, node.right], keywords=[]), node)
            return node
    tree = ReplaceProducts().visit(ast.parse(inspect.getsource(mixture.fit_spectral_mixture)))
    context = dict(mixture.__dict__, einsum_matmul=einsum_matmul)
    exec(compile(ast.fix_missing_locations(tree), "<isolated-einsum-fit>", "exec"), context)
    return context["fit_spectral_mixture"]


def audit_case(mixture, predictions, target):
    prepared = [mixture._unit_tic(p) for p in predictions if p is not None and len(p)]
    prepared_target = mixture._unit_tic(mixture.prepare_spectrum(
        mixture._unit_tic(target), relative_threshold=0, max_peaks=None, aggregation="sum"))
    masses, matrix = mixture._prediction_axis(prepared)
    pairs = mixture.match_peaks(np.column_stack((masses, np.ones(len(masses)))), prepared_target)
    aligned = np.zeros(len(masses))
    used = set()
    for i, j in pairs:
        aligned[i] = prepared_target[j, 1]
        used.add(j)
    unmatched = [prepared_target[i, 1] for i in range(len(prepared_target)) if i not in used]
    if unmatched:
        matrix = np.vstack((matrix, np.zeros((len(unmatched), len(prepared)))))
        aligned = np.concatenate((aligned, unmatched))

    original, original_warnings = captured(lambda: mixture.fit_spectral_mixture(predictions, target))
    alternate, alternate_warnings = captured(lambda: shadow_fit(mixture)(predictions, target))
    repeated, repeated_warnings = captured(lambda: mixture.fit_spectral_mixture(predictions, target))
    usable_weights = original["weights"][original["usable_indices"]]
    checks = {}
    for name, left, right in (("single_score_numerators", matrix.T, aligned),
                              ("gram_matrix", matrix.T, matrix),
                              ("mixed_intensities", matrix, usable_weights)):
        actual, operation_warnings = captured(lambda: left @ right)
        reference, reference_warnings = captured(lambda: einsum_matmul(left, right))
        checks[name] = dict(comparison=compare(actual, reference), warnings=operation_warnings,
                            einsum_warnings=reference_warnings, output=array_summary(actual))
    return dict(original_fit=original, original_warnings=original_warnings,
        shadow_fit=alternate, shadow_warnings=alternate_warnings, repeat_warnings=repeated_warnings,
        shadow_scope="Only explicit @ in fit replaced; SciPy NNLS and NumPy norm/dot unchanged.",
        original_vs_shadow={key: compare(original[key], alternate[key]) for key in
                            ("weights", "spectrum", "cosine", "best_single_cosine")},
        original_vs_repeat={key: compare(original[key], repeated[key]) for key in
                            ("weights", "spectrum", "cosine", "best_single_cosine")},
        matrix=array_summary(matrix), aligned_target=array_summary(aligned),
        matrix_column_sums=matrix.sum(axis=0).tolist(), target_sum=float(aligned.sum()),
        column_norms=np.linalg.norm(matrix, axis=0).tolist(), target_norm=float(np.linalg.norm(aligned)),
        products=checks), dict(matrix=matrix, aligned_target=aligned, masses=masses)


def main():
    ns = namespace()
    from magma_workflow import plotting
    ns.update({name: getattr(plotting, name) for name in
               ("plot_fragments_with_spectra", "_gallery_matches_ms3_energy", "_gallery_peaks")})
    from src.spectra import mixture
    original = json.loads((OUT / "spectrum_03.json").read_text())
    settings = json.loads((OUT / "run_settings.json").read_text())
    model = ns["load_glacier_model"](aromatic_fix=settings["aromatic_fix"], radical_fix=settings["radical_fix"])
    assert model.checkpoint_identity == settings["checkpoint"]
    root, scans = load_scans(original)
    bags = ns["filter_same_formula_bags"](ns["get_candidate_bags"](root, scans))
    indexed = [(i, bag) for i, bag in enumerate(bags) if bag[2]]
    chosen = [bag for _, bag in indexed]
    with budget(90):
        predictions = ns["predict_candidate_bags"](model, chosen,
            collision_energy=settings["collision_energy"], max_states=settings["max_states"])
    expected = {(row["bag_index"], row["candidate_index"]): row for row in original["candidate_rows"]}
    arrays, cases, prediction_checks = {}, [], []
    for (bag_index, (mz, candidates, observations)), (_, spectra, _) in zip(indexed, predictions):
        for ci, spectrum in enumerate(spectra):
            if spectrum is not None:
                arrays[f"bag{bag_index}_prediction{ci}"] = spectrum
                row = expected[(bag_index, ci)]
                exact = hashlib.sha256(spectrum.tobytes()).hexdigest() == row["prediction_sha256"]
                assert exact
                prediction_checks.append(dict(bag_index=bag_index, candidate_index=ci,
                    candidate_id=candidates[ci].candidate_id, exact=True, peaks=array_summary(spectrum)))
        selected = [obs for obs in observations if ns["_gallery_matches_ms3_energy"](obs, (60.,))]
        for target_index, observed in enumerate(selected):
            target = ns["_gallery_peaks"](observed, bag_index, target_index)
            prefix = f"bag{bag_index}_target{target_index}"
            arrays[prefix] = target
            if sum(s is not None and len(s) > 0 for s in spectra) < 2:
                continue
            report, internal = audit_case(mixture, spectra, target)
            arrays.update({prefix + "_" + key: value for key, value in internal.items()})
            cases.append(dict(bag_index=bag_index, mz=mz, target_index=target_index,
                target_metadata=observed.metadata, target=array_summary(target), **report))
    # Also capture the complete gallery path, which runs unrelated operations
    # before/after fitting and can alter floating-point status flags.
    html, gallery_warnings = captured(lambda: ns["plot_fragments_with_spectra"](
        chosen, predictions, ms3_collision_energies=[60], only_with_ms3=True, fit_mixture=True).data)
    config = io.StringIO()
    with redirect_stdout(config):
        np.show_config()
    report = dict(spectrum_index=3, numpy_version=np.__version__, scipy_version=scipy.__version__,
        platform=platform.platform(), numpy_configuration=config.getvalue(),
        model_settings=model.recovery_settings, collision_energy=settings["collision_energy"],
        max_states=settings["max_states"], prediction_checks=prediction_checks,
        cases=cases, gallery_warnings=gallery_warnings, gallery_html_bytes=len(html.encode()))
    np.savez_compressed(OUT / "mixture_audit_inputs.npz", **arrays)
    write_json(OUT / "mixture_audit.json", report)
    print(json.dumps(dict(cases=len(cases), exact_predictions=len(prediction_checks),
                         fit_warning_counts=[len(case["original_warnings"]) for case in cases],
                         gallery_warnings=len(gallery_warnings))), flush=True)


if __name__ == "__main__":
    main()
