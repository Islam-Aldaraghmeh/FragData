"""Drawings of molecules and MAGMa candidate fragments for notebooks."""
from .candidates import _unpack_bag
from .chemistry import formula_counts


def plot_smiles(smiles, *, size=(400, 300), legend=""):
    """Return a molecule image from SMILES; Jupyter displays it automatically.

    Example: plot_smiles("CCO") or plot_smiles(spectrum.smiles).
    For unsanitized MAGMa fragment graphs, use plot_fragment_bags instead.
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem import Draw

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("Provide a nonempty SMILES string")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"RDKit could not parse this SMILES: {smiles}")
    return Draw.MolToImage(molecule, size=size, legend=legend)


def _fragment_graph_svg(fragment, size, show_atom_indices):
    """Draw a fresh heavy graph without sanitizing, kekulizing, or adding H."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    graph = fragment.graph
    if not graph.get("atoms"):
        raise ValueError("Fragment has no atoms")
    bond_types = {1.0: Chem.BondType.SINGLE, 1.5: Chem.BondType.AROMATIC,
                  2.0: Chem.BondType.DOUBLE, 3.0: Chem.BondType.TRIPLE}
    # Block RDKit log noise; a malformed record becomes a visible error card.
    with rdBase.BlockLogs():
        editable = Chem.RWMol()
        for record in graph["atoms"]:
            atom = Chem.Atom(record["element"])
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
            atom.SetNumRadicalElectrons(0)
            atom.SetIsAromatic(bool(record.get("aromatic", False)))
            editable.AddAtom(atom)
        for record in graph["bonds"]:
            order = float(record["order"])
            if order not in bond_types:
                raise ValueError(f"Unsupported bond order: {order}")
            i, j = int(record["i"]), int(record["j"])
            editable.AddBond(i, j, bond_types[order])
            editable.GetBondBetweenAtoms(i, j).SetIsAromatic(
                bool(record.get("aromatic", False)) or order == 1.5
            )
        molecule = editable.GetMol()
        molecule.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(molecule)
        rdDepictor.Compute2DCoords(molecule)

        drawer = rdMolDraw2D.MolDraw2DSVG(*size)
        options = drawer.drawOptions()
        options.prepareMolsBeforeDrawing = False
        options.includeRadicals = False
        options.addAtomIndices = show_atom_indices
        for atom in molecule.GetAtoms():
            if atom.GetDegree() == 0:
                options.atomLabels[atom.GetIdx()] = atom.GetSymbol()
        drawer.DrawMolecule(molecule)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
    return svg[svg.index("<svg"):]


def plot_fragment_bags(candidate_bags, *, size=(260, 200), show_atom_indices=False):
    """Return notebook HTML: one row per (m/z, fragments, ms3_spectra) bag.

    All candidates retain input order; long rows scroll rather than wrap or
    truncate. Draw only heavy connectivity/bond orders, with formula/charge in
    captions. Imperfect valence/aromaticity does not require chemistry repair.
    Malformed graph records keep an error card. Source candidates are untouched.
    Pass either a list of bags or a single bag such as bags[i]. Use
    show_atom_indices=True to label the graph's local atom indices.
    """
    import math
    from numbers import Real
    from html import escape
    from IPython.display import HTML
    from rdkit import Chem  # Fail once if the notebook kernel lacks RDKit.

    if (len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                              for v in size)):
        raise ValueError("size must be (width, height) in positive integer pixels")
    width, height = size
    # Also accept the single (m/z, fragments) pair returned by bags[i].
    if (isinstance(candidate_bags, (tuple, list)) and len(candidate_bags) in (2, 3)
            and isinstance(candidate_bags[0], Real)):
        candidate_bags = [candidate_bags]
    sections = []
    for bag_index, bag in enumerate(candidate_bags):
        mz, fragments, _ = _unpack_bag(bag)
        mz = float(mz)
        if not math.isfinite(mz) or mz <= 0:
            raise ValueError(f"Bag {bag_index} has an invalid m/z")
        fragments = list(fragments)
        cards = []
        for fragment_index, fragment in enumerate(fragments):
            try:
                counts = formula_counts(fragment.ionic_formula)
                elements = ([e for e in ("C", "H") if e in counts]
                            if "C" in counts else [])
                elements += sorted(set(counts) - set(elements))
                formula = "".join(e + (str(counts[e]) if counts[e] != 1 else "")
                                  for e in elements)
            except (AttributeError, TypeError, ValueError):
                formula = str(getattr(fragment, "ionic_formula", "formula unavailable"))
            charge = getattr(fragment, "charge", "?")
            charge_text = f"{charge:+d}" if isinstance(charge, int) else str(charge)
            try:
                drawing = _fragment_graph_svg(fragment, (width, height), show_atom_indices)
            except (AttributeError, KeyError, IndexError, TypeError, ValueError,
                    RuntimeError, OverflowError) as error:
                # Keep this candidate's position without breaking the other drawings.
                reason = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
                drawing = (
                    f'<div style="height:{height}px;display:flex;align-items:center;'
                    'justify-content:center;padding:12px;box-sizing:border-box;color:#9b3535;'
                    'overflow:auto"><div><b>Graph unavailable</b><br>'
                    f'<small>{escape(reason)}</small></div></div>'
                )
            caption = (f'<b>Candidate {fragment_index}</b><br>{escape(formula)}'
                       f' &nbsp;|&nbsp; charge {escape(charge_text)}')
            cards.append(
                f'<div style="flex:0 0 {width}px;width:{width}px;border:1px solid #ddd;'
                'border-radius:6px;overflow:hidden;background:white;color:#222">'
                f'{drawing}<div style="padding:8px;text-align:center;font-size:13px">'
                f'{caption}</div></div>'
            )
        row = (''.join(cards) if cards else
               '<div style="padding:16px;color:#666">No candidate fragments</div>')
        sections.append(
            '<section style="margin:0 0 18px">'
            f'<div style="margin-bottom:8px"><b>Bag {bag_index} · m/z {mz:.5f}</b>'
            f' &nbsp;({len(fragments)} candidates)</div>'
            '<div style="display:flex;flex-wrap:nowrap;gap:10px;overflow-x:auto;'
            f'padding-bottom:8px">{row}</div></section>'
        )
    if not sections:
        return HTML('<p>No candidate bags to display.</p>')
    return HTML(
        '<div style="max-width:100%;font-family:system-ui,sans-serif">'
        '<p style="font-size:13px">Heavy-atom sketches; hydrogen and charge positions '
        'are not assigned. Scroll sideways to see every candidate.</p>'
        + ''.join(sections) + '</div>'
    )


def plot_candidate_rows(fragments, mz=None, spectra=None, captions=None, *, measured=None,
                        order=None, size=(240, 170)):
    """Return notebook HTML: one row per candidate with its sketch, caption and optional spectra.

    ``spectra[k]`` holds candidate k's heights at the shared ``mz`` values; every plot
    uses the same axes. ``captions[k]`` (a string or a list of lines) is shown under
    "Candidate k". ``measured`` adds a last column with the measured heights at ``mz``
    (the same in every row). ``order`` lists the candidate indices top to bottom.
    """
    import base64
    import io
    from html import escape

    import matplotlib.pyplot as plt
    from IPython.display import HTML

    fragments = list(fragments)
    top = max([float(max(s)) for s in spectra] if spectra is not None else [0.0])
    top = max(top, float(max(measured)) if measured is not None else 0.0) or 1.0

    def chart(heights, label, color):
        figure, ax = plt.subplots(figsize=(5.2, size[1] / 100))
        ax.vlines(mz, 0, heights, color=color, linewidth=1.3)
        ax.set_xlim(min(mz) - 5, max(mz) + 5)
        ax.set_ylim(0, top * 1.05)
        ax.set_xlabel("m/z")
        ax.set_ylabel(label)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        plt.close(figure)
        return '<td><img src="data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode() + '"></td>'

    measured_cell = chart(measured, "measured MS3", "#1f4e9c") if measured is not None else ""
    rows = []
    for k in (range(len(fragments)) if order is None else order):
        lines = [] if captions is None else ([captions[k]] if isinstance(captions[k], str) else list(captions[k]))
        cells = [f'<td style="width:{size[0]}px">{_fragment_graph_svg(fragments[k], size, False)}</td>',
                 f'<td style="padding:0 14px;white-space:nowrap"><b>Candidate {k}</b>'
                 + ''.join(f'<br>{escape(str(line))}' for line in lines) + '</td>']
        if spectra is not None:
            cells.append(chart(spectra[k], "predicted", "black"))
        rows.append('<tr style="border-bottom:1px solid #ddd;background:white;color:#222">'
                    + ''.join(cells) + measured_cell + '</tr>')
    return HTML('<table style="border-collapse:collapse;font-family:system-ui,sans-serif;font-size:13px">'
                + ''.join(rows) + '</table>')
