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
