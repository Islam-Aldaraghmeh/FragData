"""Read-only diagnostics of the saved intensity model and feature cache."""
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from intensity_model import FP_SIZE, normalize_intensity, pool_candidate_scores, fingerprint_molecule, structure_key
from train_intensity import load_model, predict_examples


def cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 0 else 0.


def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(np.mean(a)), median=float(np.median(a)),
                q10=float(np.quantile(a, .1)), q90=float(np.quantile(a, .9))) if len(a) else {}


def correlate(a, b):
    return float(spearmanr(a, b).statistic) if len(a) >= 3 else None


def mass_match(a, b):
    """One-to-one nearest eligible m/z pairs, .005 Da or 20 ppm."""
    candidates = []
    for i, mz in enumerate(a):
        tolerance = max(.005, 20e-6 * mz)
        for j in np.flatnonzero(np.abs(b - mz) <= tolerance):
            candidates.append((abs(float(b[j] - mz)), i, int(j)))
    left, right, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in left and j not in right:
            pairs.append((i, j))
            left.add(i)
            right.add(j)
    return np.asarray(sorted(pairs), dtype=int).reshape(-1, 2)


def bag_feature_key(example, peak):
    rows = example['features'][example['peak_indices'] == peak].copy()
    rows[:, 2 * FP_SIZE] = 0
    return tuple(sorted(row.tobytes() for row in rows))


def run():
    torch.set_num_threads(2)
    folder = ROOT / 'results' / 'experiment_2'
    with (folder / 'dataset.pkl').open('rb') as f:
        examples = pickle.load(f)
    model = load_model(folder / 'model.pt')
    predictions = predict_examples(model, examples)
    groups = defaultdict(list)
    audits = []
    collision_totals = Counter()
    duplicate_examples = []
    indistinguishable_examples = []
    for index, (example, pred) in enumerate(zip(examples, predictions)):
        groups[example['molecule']].append(index)
        support = np.bincount(example['peak_indices'], minlength=len(example['target']))
        collision_totals['candidate_rows'] += len(example['features'])
        collision_totals['supported_peaks'] += int(np.sum(support > 0))
        peak_keys = defaultdict(list)
        for peak in np.flatnonzero(support):
            rows = example['features'][example['peak_indices'] == peak]
            keys = [row.tobytes() for row in rows]
            duplicates = len(keys) - len(set(keys))
            collision_totals['within_peak_duplicate_feature_rows'] += duplicates
            collision_totals['peaks_with_duplicate_features'] += int(duplicates > 0)
            peak_keys[tuple(sorted(set(keys)))].append(int(peak))
            if duplicates and len(duplicate_examples) < 10:
                duplicate_examples.append(dict(source_offset=example['source_offset'],
                    mz=float(example['mz'][peak]), candidates=len(keys), duplicate_rows=duplicates))
        indistinguishable = [v for v in peak_keys.values() if len(v) > 1]
        collision_totals['indistinguishable_peak_groups'] += len(indistinguishable)
        collision_totals['indistinguishable_peak_rows'] += sum(map(len, indistinguishable))
        constrained_oracle = example['target'].copy()
        constrained_oracle[support == 0] = 0
        support_oracle = constrained_oracle.copy()
        for same_peaks in indistinguishable:
            constrained_oracle[same_peaks] = np.mean(example['target'][same_peaks])
            indistinguishable_examples.append(dict(source_offset=example['source_offset'],
                split=example['split'], masses=example['mz'][same_peaks].tolist(),
                targets=example['target'][same_peaks].tolist(),
                gap=float(np.ptp(example['mz'][same_peaks])),
                target_gap=float(np.ptp(example['target'][same_peaks]))))
        audits.append(dict(index=index, split=example['split'], molecule=example['molecule'],
            source_offset=example['source_offset'], ce=example['collision_energy'],
            cosine=cosine(example['target'], pred),
            supported_cosine=cosine(example['target'][support > 0], pred[support > 0]),
            supported_peak_count=int(np.sum(support > 0)),
            mean_candidates=float(support[support > 0].mean()) if np.any(support) else 0,
            max_candidates=int(support.max(initial=0)),
            indistinguishable_peak_groups=len(indistinguishable),
            support_ceiling=cosine(example['target'], support_oracle),
            feature_ceiling=cosine(example['target'], constrained_oracle),
            ambiguous_fraction=float(np.mean(support[support > 0] > 1)) if np.any(support) else 0))
    counterfactual = []
    for index, example in enumerate(examples):
        if not len(example['features']):
            continue
        indices = torch.as_tensor(example['peak_indices'])
        spectra = []
        for ce in (20., 60.):
            features = example['features'].copy()
            features[:, 2 * FP_SIZE] = ce / 100
            with torch.no_grad():
                candidate = model(torch.as_tensor(features))
                pooled = pool_candidate_scores(candidate, indices, len(example['target'])).numpy()
            spectra.append(normalize_intensity(pooled))
        counterfactual.append(dict(index=index, split=example['split'],
            ce20_vs60_cosine=cosine(spectra[0], spectra[1]),
            ce20_vs60_mean_abs_change=float(np.mean(np.abs(spectra[0] - spectra[1]))),
            ce20_vs60_max_abs_change=float(np.max(np.abs(spectra[0] - spectra[1])))))
    pairs = []
    for molecule, group in groups.items():
        for a, b in combinations(group, 2):
            ea, eb = examples[a], examples[b]
            matched = mass_match(ea['mz'], eb['mz'])
            if len(matched) < 5:
                continue
            ia, ib = matched.T
            ya, yb = ea['target'][ia], eb['target'][ib]
            pa, pb = predictions[a][ia], predictions[b][ib]
            supported = (pa > 0) & (pb > 0)
            if np.sum(supported) < 5:
                continue
            shared_ia, shared_ib = ia[supported], ib[supported]
            identical = [bag_feature_key(ea, i) == bag_feature_key(eb, j)
                         for i, j in zip(shared_ia, shared_ib)]
            pairs.append(dict(molecule=molecule, split=ea['split'], a=a, b=b,
                ce_a=ea['collision_energy'], ce_b=eb['collision_energy'],
                source_offset_a=ea['source_offset'], source_offset_b=eb['source_offset'],
                same_precursor_smiles=ea['smiles'] == eb['smiles'],
                matched_peaks=len(matched), shared_supported_peaks=int(np.sum(supported)),
                identical_feature_bags=int(sum(identical)),
                squared_intensity_retained_a=float(np.sum(ya[supported] ** 2) / np.sum(ea['target'] ** 2)),
                squared_intensity_retained_b=float(np.sum(yb[supported] ** 2) / np.sum(eb['target'] ** 2)),
                measured_cosine_shared=cosine(ya[supported], yb[supported]),
                predicted_cosine_shared=cosine(pa[supported], pb[supported]),
                measured_mean_abs_change=float(np.mean(np.abs(ya[supported] - yb[supported]))),
                predicted_mean_abs_change=float(np.mean(np.abs(pa[supported] - pb[supported])))))
    split_summary = {}
    for split in ('train', 'validation', 'test'):
        a = [e for e in audits if e['split'] == split]
        split_summary[split] = dict(
            ce_counts=dict(Counter(str(e['ce']) for e in a)),
            mean_candidates=describe([e['mean_candidates'] for e in a]),
            ambiguous_fraction=describe([e['ambiguous_fraction'] for e in a]),
            spectra_with_indistinguishable_peaks=sum(e['indistinguishable_peak_groups'] > 0 for e in a),
            mean_support_ceiling=float(np.mean([e['support_ceiling'] for e in a])),
            mean_feature_ceiling=float(np.mean([e['feature_ceiling'] for e in a])),
            ambiguity_cosine_spearman=correlate([e['mean_candidates'] for e in a], [e['cosine'] for e in a]),
            ambiguity_supported_cosine_spearman=correlate([e['mean_candidates'] for e in a], [e['supported_cosine'] for e in a]),
            by_ce={str(ce): dict(n=len(g), mean_cosine=float(np.mean([e['cosine'] for e in g])),
                                mean_supported_cosine=float(np.mean([e['supported_cosine'] for e in g])))
                   for ce in sorted({e['ce'] for e in a})
                   for g in [[e for e in a if e['ce'] == ce]]})
    counterfactual_summary = {}
    pair_summary = {}
    for split in ('train', 'validation', 'test'):
        group = [r for r in counterfactual if r['split'] == split]
        counterfactual_summary[split] = {k: describe([r[k] for r in group])
             for k in ('ce20_vs60_cosine', 'ce20_vs60_mean_abs_change', 'ce20_vs60_max_abs_change')}
        group = [r for r in pairs if r['split'] == split]
        pair_summary[split] = {k: describe([r[k] for r in group])
             for k in ('matched_peaks', 'shared_supported_peaks', 'measured_cosine_shared',
                       'predicted_cosine_shared', 'measured_mean_abs_change', 'predicted_mean_abs_change',
                       'squared_intensity_retained_a', 'squared_intensity_retained_b')}
        pair_summary[split]['pairs_with_all_identical_bags'] = sum(r['identical_feature_bags'] == r['shared_supported_peaks'] for r in group)
        pair_summary[split]['identical_bag_fraction'] = sum(r['identical_feature_bags'] for r in group) / sum(r['shared_supported_peaks'] for r in group)
        pair_summary[split]['pairs_same_precursor_smiles'] = sum(r['same_precursor_smiles'] for r in group)
    ce_examples = []
    eligible = sorted([r for r in pairs if r['split'] == 'test' and set((r['ce_a'], r['ce_b'])) == {20., 60.}],
                      key=lambda r: r['measured_cosine_shared'])
    for record in [eligible[0], eligible[len(eligible)//2]] if eligible else []:
        a, b = record['a'], record['b']
        if examples[a]['collision_energy'] > examples[b]['collision_energy']:
            a, b = b, a
        ea, eb = examples[a], examples[b]
        ia, ib = mass_match(ea['mz'], eb['mz']).T
        supported = (predictions[a][ia] > 0) & (predictions[b][ib] > 0)
        ia, ib = ia[supported], ib[supported]
        ce_examples.append(dict(record=record, source_offset_20=ea['source_offset'], source_offset_60=eb['source_offset'],
            mz20=ea['mz'][ia].tolist(), mz60=eb['mz'][ib].tolist(),
            target20=ea['target'][ia].tolist(), target60=eb['target'][ib].tolist(),
            prediction20=predictions[a][ia].tolist(), prediction60=predictions[b][ib].tolist()))
    result = dict(collision_totals=dict(collision_totals), duplicate_examples=duplicate_examples,
                  indistinguishable_examples=sorted(indistinguishable_examples,
                    key=lambda e: e['target_gap'], reverse=True),
                  split_summary=split_summary,
                  counterfactual_summary=counterfactual_summary,
                  matched_ce_pair_summary=pair_summary, ce_examples=ce_examples,
                  spectrum_audits=audits, counterfactual=counterfactual, matched_ce_pairs=pairs)
    training_molecules = sorted({e['molecule'] for e in examples if e['split'] == 'train'})
    training_fp = np.asarray([fingerprint_molecule(m) for m in training_molecules])
    nearest = []
    for molecule in sorted(groups):
        index = groups[molecule][0]
        split = examples[index]['split']
        if split == 'train':
            continue
        fp = fingerprint_molecule(molecule)
        intersection = np.sum(training_fp * fp, axis=1)
        similarity = intersection / (training_fp.sum(axis=1) + fp.sum() - intersection)
        best = int(np.argmax(similarity))
        nearest.append(dict(molecule=molecule, split=split, nearest_train=training_molecules[best],
            nearest_model_fingerprint_tanimoto=float(similarity[best]),
            mean_cosine=float(np.mean([audits[i]['cosine'] for i in groups[molecule]]))))
    result['nearest_parent'] = nearest
    result['nearest_parent_summary'] = {split: dict(
        max_train_tanimoto=describe([r['nearest_model_fingerprint_tanimoto'] for r in nearest if r['split'] == split]),
        similarity_vs_cosine_spearman=correlate(
            [r['nearest_model_fingerprint_tanimoto'] for r in nearest if r['split'] == split],
            [r['mean_cosine'] for r in nearest if r['split'] == split])) for split in ('validation', 'test')}
    # Confirm a few tied-peak examples share actual exact structures, not just FP bits.
    from magma_workflow.candidates import get_candidate_bags
    from magma_workflow.dataset import Spectrum
    from intensity_data import GENERATION
    manifest = json.loads((folder / 'manifest.json').read_text())
    tied_verification = []
    with Path(manifest['cache_key']['source']).open('rb') as source:
        chosen_offsets = set()
        for case in result['indistinguishable_examples']:
            if case['split'] != 'test' or case['source_offset'] in chosen_offsets:
                continue
            source.seek(case['source_offset'])
            raw = Spectrum.from_record(json.loads(source.readline()))
            placeholder = Spectrum(raw.smiles, raw.precursor_mz, 2,
                np.column_stack((raw.mz, np.zeros(len(raw.mz)))), raw.metadata.copy())
            bags = get_candidate_bags(placeholder, spectra=[], **GENERATION)
            chosen_bags = [bags[int(np.argmin(np.abs(raw.mz - mz)))] for mz in case['masses']]
            keys = [sorted(structure_key(f) for f in bag[1]) for bag in chosen_bags]
            tied_verification.append(dict(case, exact_structure_sets_equal=all(k == keys[0] for k in keys),
                candidate_counts=[len(bag[1]) for bag in chosen_bags],
                candidate_ids=[[f.candidate_id for f in bag[1]] for bag in chosen_bags],
                theoretical_masses=[sorted({f.metadata['theoretical_mz'] for f in bag[1]}) for bag in chosen_bags]))
            if not chosen_offsets:
                original_index = next(i for i, e in enumerate(examples) if e['source_offset'] == case['source_offset'])
                e = examples[original_index]
                p = predictions[original_index]
                rows = []
                for mz in case['masses']:
                    peak = int(np.argmin(np.abs(e['mz'] - mz)))
                    theory = float(chosen_bags[0][1][0].metadata['theoretical_mz'])
                    rows.append(dict(observed_mz=mz, theoretical_mz=theory,
                        signed_mass_error_da=mz-theory, signed_mass_error_ppm=1e6*(mz-theory)/theory,
                        target=float(e['target'][peak]), prediction=float(p[peak])))
                tie_case = dict(source_offset=case['source_offset'], split=case['split'],
                    smiles=e['smiles'], collision_energy=e['collision_energy'],
                    exact_structure_sets_equal=all(k == keys[0] for k in keys),
                    candidate_ids=[[f.candidate_id for f in bag[1]] for bag in chosen_bags],
                    candidate_graph=chosen_bags[0][1][0].graph,
                    ionic_formula=chosen_bags[0][1][0].ionic_formula, rows=rows)
                (folder / 'error_analysis' / 'tiebag_example.json').write_text(json.dumps(tie_case, indent=2) + '\n')
            chosen_offsets.add(case['source_offset'])
            if len(chosen_offsets) == 3:
                break
    result['tied_peak_exact_structure_verification'] = tied_verification
    out = folder / 'error_analysis' / 'feature_audit.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('spectrum_audits', 'counterfactual', 'matched_ce_pairs', 'indistinguishable_examples', 'ce_examples', 'nearest_parent')}, indent=2))


if __name__ == '__main__':
    run()
