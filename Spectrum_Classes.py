from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt


class SpectrumCollection(list):
    """A normal list with a constant-size preview for notebook inspectors."""
    __slots__ = ()

    def __repr__(self):
        return f"SpectrumCollection({len(self)} spectra)"

    def _repr_pretty_(self, printer, cycle):
        printer.text(repr(self))


class SpectraBySmiles(dict):
    """A normal mapping whose preview does not render the entire library."""
    __slots__ = ()

    def __repr__(self):
        return f"SpectraBySmiles({len(self)} SMILES groups)"

    def _repr_pretty_(self, printer, cycle):
        printer.text(repr(self))

@dataclass(slots=True)
class Spectrum:
    smiles: str
    precursor_mz: float
    ms_level: int
    peaks: np.ndarray
    metadata: dict

    def __repr__(self):
        # Dataclass defaults render every peak and metadata entry. IDE variable
        # previews can request this after every cell, including a plain print().
        return (f"Spectrum(ms_level={self.ms_level}, "
                f"precursor_mz={self.precursor_mz:.6f}, n_peaks={len(self.peaks)})")

    @classmethod
    def from_record(cls, record):
        data = record.copy()
        return cls(
            smiles=data.pop("smiles"),
            precursor_mz=data.pop("precursor_mz"),
            ms_level=data.pop("ms_level"),
            peaks=np.asarray(data.pop("peaks"), dtype=float).reshape(-1, 2),
            metadata=data,
        )

    @property
    def mz(self):
        return self.peaks[:, 0]

    @property
    def intensity(self):
        return self.peaks[:, 1]

    def plot(self, normalize=True, figsize=(10, 4)):

        intensities = self.intensity.copy()
        if normalize and intensities.size and intensities.max() > 0:
            intensities = 100 * intensities / intensities.max()

        _, ax = plt.subplots(figsize=figsize)
        ax.vlines(self.mz, 0, intensities, color="black", linewidth=1)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Relative intensity (%)" if normalize else "Intensity")

        name = self.metadata.get("compound_name", self.smiles)
        scan_numbers = self.metadata.get("scan_number")
        if isinstance(scan_numbers, (list, tuple, np.ndarray)):
            scan = scan_numbers[0] if len(scan_numbers) else None
        else:
            scan = scan_numbers
        merge_type = self.metadata.get("merge_type")
        title = f"{name} | MS{self.ms_level} | precursor m/z {self.precursor_mz:.4f}"
        if scan is not None:
            title += f" | scan {scan}"
        if merge_type:
            title += f" [{merge_type}]"
        title +=f" {self.metadata['collision_energy']}ev"
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()

    def plot_root(self, size=(400, 300)):
        from IPython.display import display
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
        except ImportError as error:
            raise ImportError(
                "plot_root requires RDKit to be installed in the notebook kernel"
            ) from error

        molecule = Chem.MolFromSmiles(self.smiles)
        if molecule is None:
            raise ValueError(f"RDKit could not parse this SMILES: {self.smiles}")

        name = self.metadata.get("compound_name", "")
        image = Draw.MolToImage(molecule, size=size, legend=name)
        display(image)


@dataclass(slots=True)
class SpectrumForest:
    nodes: dict[str, Spectrum]
    parent_of: dict[str, str] = field(default_factory=dict)
    parent_peak_mz: dict[str, float] = field(default_factory=dict)
    orphans: list[str] = field(default_factory=list)

    def __repr__(self):
        return (f"SpectrumForest(n_spectra={len(self.nodes)}, "
                f"n_links={len(self.parent_of)}, n_orphans={len(self.orphans)})")

    @property
    def roots(self):
        return [s for s in self.nodes.values() if s.ms_level == 2]

    def parent(self, scan):
        return self.nodes.get(self.parent_of.get(scan))

    def children(self, scan):
        return [self.nodes[c] for c, p in self.parent_of.items() if p == scan]

    @staticmethod
    def _scan_number(spectrum):
        scan = spectrum.metadata.get("scan_number")
        return scan[0] if isinstance(scan, (list, tuple)) and scan else scan

    def plot_root(self, size=(400, 300)):
        if not self.nodes:
            raise ValueError("Cannot plot the root molecule of an empty forest")

        smiles = {spectrum.smiles for spectrum in self.nodes.values()}
        if len(smiles) != 1:
            raise ValueError(
                f"Forest contains {len(smiles)} different molecules; expected one SMILES"
            )

        return next(iter(self.nodes.values())).plot_root(size=size)

    def _spectrum_for_scan(self, scan_number):
        if scan_number is None:
            raise ValueError("scan_number is required")

        matches = [s for s in self.nodes.values()
                   if self._scan_number(s) == scan_number]
        if not matches:
            raise KeyError(f"No spectrum found for scan {scan_number}")
        if len(matches) > 1:
            feature_ids = sorted({s.metadata["feature_id"] for s in matches})
            raise ValueError(
                f"Scan {scan_number} occurs in multiple feature/source files: "
                f"{feature_ids}. Build a forest for one feature before plotting."
            )

        return matches[0]

    def plot_spectrum(self, scan_number=None, **plot_kwargs):
        return self._spectrum_for_scan(scan_number).plot(**plot_kwargs)

    def plot_multiple(self, scan_numbers, dimensions=None, normalize=True, figsize=None):
        import matplotlib.pyplot as plt

        scan_numbers = list(scan_numbers)
        if not scan_numbers:
            raise ValueError("scan_numbers must contain at least one scan")
        spectra = [self._spectrum_for_scan(scan) for scan in scan_numbers]

        if dimensions is None:
            dimensions = (len(spectra), 1)
        if (not isinstance(dimensions, tuple) or len(dimensions) != 2
                or any(not isinstance(size, int) or size <= 0 for size in dimensions)):
            raise ValueError("dimensions must be a (rows, columns) tuple of positive integers")

        rows, columns = dimensions
        if rows * columns < len(spectra):
            raise ValueError(
                f"Grid {dimensions} has {rows * columns} slots for {len(spectra)} scans"
            )
        if figsize is None:
            figsize = (6 * columns, 3 * rows)

        plotted_intensities = []
        for spectrum in spectra:
            intensities = spectrum.intensity.copy()
            if normalize and intensities.size and intensities.max() > 0:
                intensities = 100 * intensities / intensities.max()
            plotted_intensities.append(intensities)

        nonempty_mz = [spectrum.mz for spectrum in spectra if spectrum.mz.size]
        if not nonempty_mz:
            raise ValueError("The selected spectra contain no peaks")
        all_mz = np.concatenate(nonempty_mz)
        mz_min, mz_max = float(all_mz.min()), float(all_mz.max())
        mz_padding = max((mz_max - mz_min) * 0.02, 1.0)
        if normalize:
            intensity_max = 100.0
        else:
            intensity_max = max(
                (float(values.max()) for values in plotted_intensities if values.size),
                default=1.0,
            )

        figure, axes = plt.subplots(
            rows, columns, figsize=figsize, squeeze=False, sharex=True, sharey=True
        )
        flat_axes = axes.ravel()
        for ax, scan, spectrum, intensities in zip(
                flat_axes, scan_numbers, spectra, plotted_intensities):
            ax.vlines(spectrum.mz, 0, intensities, color="black", linewidth=1)
            ax.set_xlim(mz_min - mz_padding, mz_max + mz_padding)
            ax.set_ylim(0, intensity_max * 1.05 if intensity_max > 0 else 1)
            ax.set_xlabel("m/z")
            ax.set_ylabel("Relative intensity (%)" if normalize else "Intensity")
            ax.tick_params(labelbottom=True, labelleft=True)
            merge_type = spectrum.metadata.get("merge_type", "unknown")
            ce = spectrum.metadata.get("collision_energy", "?")
            ax.set_title(f"MS{spectrum.ms_level} | scan {scan} | CE={ce} [{merge_type}]")

        for ax in flat_axes[len(spectra):]:
            ax.axis("off")

        figure.tight_layout()
        plt.show()

    def show_forest_summary(self):
        feature_ids = sorted({s.metadata["feature_id"] for s in self.nodes.values()})
        print(
            f"{len(feature_ids)} feature forest(s) | {len(self.nodes)} spectra | "
            f"{len(self.parent_of)} parent-child links | {len(self.roots)} MS2 roots | "
            f"{len(self.orphans)} unresolved MS3+ scans"
        )

        for feature_id in feature_ids:
            spectra = [s for s in self.nodes.values()
                       if s.metadata["feature_id"] == feature_id]
            levels = {}
            for spectrum in spectra:
                levels[spectrum.ms_level] = levels.get(spectrum.ms_level, 0) + 1
            level_text = ", ".join(
                f"MS{level}: {count}" for level, count in sorted(levels.items())
            )
            roots = [self._scan_number(s) for s in spectra if s.ms_level == 2]
            unresolved = sum(node_id in self.orphans for node_id, s in self.nodes.items()
                             if s.metadata["feature_id"] == feature_id)
            print(f"\n{feature_id}")
            print(f"  {level_text} | MS2 roots: {roots} | unresolved: {unresolved}")

    def show_forest(self):
        def visit(node_id, indent=""):
            spectrum = self.nodes[node_id]
            scan = self._scan_number(spectrum)
            ce = spectrum.metadata["collision_energy"]
            edge = (f"m/z {self.parent_peak_mz[node_id]:.5f} -> "
                    if node_id in self.parent_of else "")
            print(f"{indent}{edge}MS{spectrum.ms_level} scan {scan} (CE={ce})")
            child_ids = [child for child, parent in self.parent_of.items()
                         if parent == node_id]
            child_ids.sort(key=lambda child: self._scan_number(self.nodes[child]))
            for child_id in child_ids:
                visit(child_id, indent + "    ")

        feature_ids = sorted({s.metadata["feature_id"] for s in self.nodes.values()})
        for feature_id in feature_ids:
            roots = [s for s in self.roots if s.metadata["feature_id"] == feature_id]
            roots.sort(key=self._scan_number)
            print(f"\nFeature forest: {feature_id}")
            print("MS2 roots:", [self._scan_number(s) for s in roots])
            for root in roots:
                visit(root.metadata["usi"][0])

        print("Unresolved MS3+ scans:",
              [self._scan_number(self.nodes[node_id]) for node_id in self.orphans])
        
        
        
def parse_spectrum_forest(scans, mz_tol=0.01):
    scans = [s for s in scans if s.metadata.get("merge_type") == "SINGLE_BEST_SCAN"]
    nodes = {s.metadata["usi"][0]: s for s in scans}
    forest = SpectrumForest(nodes)

    def path(s):
        if s.ms_level == 2:
            return np.array([s.precursor_mz]), (s.metadata["collision_energy"],)
        return (np.asarray(s.metadata["msn_precursor_mzs"]),
                tuple(s.metadata["msn_collision_energies"]))

    for child_id, child in nodes.items():
        if child.ms_level == 2:
            continue
        child_mz, child_ce = path(child)
        candidates = []
        for parent_id, parent in nodes.items():
            if (parent.metadata["feature_id"] != child.metadata["feature_id"]
                    or parent.ms_level != child.ms_level - 1
                    or parent.metadata["scan_number"] >= child.metadata["scan_number"]):
                continue
            parent_mz, parent_ce = path(parent)
            if (len(parent_mz) != len(child_mz) - 1 or parent_ce != child_ce[:-1]
                    or not np.allclose(parent_mz, child_mz[:-1], atol=mz_tol, rtol=0)):
                continue
            peak_error = np.min(np.abs(parent.mz - child_mz[-1])) if parent.mz.size else np.inf
            if peak_error <= mz_tol:
                candidates.append((peak_error + np.abs(parent_mz - child_mz[:-1]).sum(), parent_id))

        if candidates:
            forest.parent_of[child_id] = min(candidates, key=lambda item: item[0])[1]
            forest.parent_peak_mz[child_id] = float(child_mz[-1])
        else:
            forest.orphans.append(child_id)
    return forest
