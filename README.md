**Predicting Biomolecule Adsorption on Multi-Phase Cement Surfaces Using PMF-Based Machine Learning**


# Abstract

Air pollution represents a major global health risk, while occupational exposure to fine particulate matter generated during cement production and construction represents an important inhalation hazard. The World Health Organization estimated that ambient outdoor air pollution caused approximately 4.2 million premature deaths worldwide in 2019. Traditional toxicological assessment methods - including in vitro cell-based assays and in vivo animal studies - are slow, expensive, and they fail to capture the chemical complexity of real-world industrial dust matrices. Conventional Quantitative Structure-Activity Relationship (QSAR) models rely on simplified descriptors such as particle size and total surface charge. These models break down when applied to irregular, multi-component materials like cement clinker. This is because the biologically relevant interactions happen at localised surface sites, and those sites differentiate across the material's composition.

This thesis presents a physics-driven computational pipeline for predicting biomolecule adsorption on the major mineral phases of industrial cement: Tricalcium Silicate (C3S/Alite), Dicalcium Silicate (C2S/Belite), Tricalcium Aluminate (C3A), and Tetracalcium Aluminoferrite (C4AF/Brownmillerite). Atomistic surface slab models were constructed from crystallographic reference structures using CHARMM-GUI Nanomaterial Modeller (for C3S and C3A) and the Crystallography Open Database with a custom pymatgen-based conversion script (for C2S and C4AF). The PMFPredictor machine-learning toolkit was used to predict Potential of Mean Force (PMF) curves describing the free energy of interaction between each mineral surface and a library of 89 biomolecules, including all 20 amino acid side chain analogues, phospholipid headgroup analogues, sugars and small organic molecules.

Two complementary approaches for generating a representative interaction profile for the complete multi-phase cement material are developed. Route 1, a weighted average of per-phase predictions using volume fractions of cement minerals. This is fully implemented and applied to three cement formulations. Route 2, a spatially patched composite surface approach, is implemented at the structural level and its technical challenges are characterised and discussed. The Route 1 pipeline is extended to compare three cement formulations: Ordinary Portland Cement (OPC), Fly Ash cement, and Slag cement.

Ensemble-averaged predictions, computed as the mean and standard deviation of adsorption energies calculated independently for each of 10 bootstrap models, reveal that phosphate-containing molecules (DMEP, PHO: -19.25 kJ/mol) are the strongest predicted binders to OPC cement, followed by sugar molecules and aromatic amino acids, including tryptophan (-14.38 kJ/mol). C4AF shows exceptional phosphate affinity (-29.50 kJ/mol ensemble mean), while C3S and C4AF show similar predicted glycine binding within the ensemble uncertainty. The multi-cement comparison reveals that Fly Ash cement, with its higher C3A content, shifts binding preference towards phosphate containing molecules relative to OPC, with implications for differential pulmonary surfactant disruption across cement formulations.

# Chapter 1: Introduction

## 1.1 Motivation and Industrial Context

The global construction industry employs a huge global workforce, many of whom are exposed daily to airborne cement particles generated during grinding, mixing, and handling of clinker. Cement dust is a chemically complex, multi-mineral particulate matrix rather than a single chemical compound, containing multiple crystalline and amorphous phases with distinct physicochemical properties and potential biological reactivities (Pietsch et al., 2021; Stutzman et al., 2015). When inhaled, fine cement particles (PM2.5 and below) penetrate deep into the alveolar regions of the lung. Here, they interact directly with biological membranes, pulmonary surfactant proteins and the cells of the innate immune system(Pietsch et al., 2021). The World Health Organization estimated that ambient outdoor air pollution caused approximately 4.2 million premature deaths worldwide in 2019, with exposure to fine particulate matter (PM2.5) identified as a major contributor (WHO, 2021), acting as a primary driver of chronic respiratory conditions including asthma, chronic obstructive pulmonary disease (COPD), pulmonary fibrosis, and systemic tissue inflammation. The AQLI's 2025 annual update estimates that permanently reducing global PM2.5 concentrations to the WHO guideline would increase average global life expectancy by approximately 1.9 years (AQLI,2026).

The toxicological response to inhaled particles depends critically on the physicochemical properties of the particle surface rather than on bulk composition alone. Surface charge distribution, elemental composition, surface area and the spatial arrangement of reactive sites all govern how a particle interacts with the biological environment. This creates a real challenge for toxicological assessment. Industrial dusts cannot be easily characterised by simple bulk descriptors. Yet the surface-level detail needed for accurate toxicity prediction has historically required expensive experiments or long computational simulations to obtain.

## 1.2 The Gap in Current Toxicological Assessment

Traditional toxicological assessment relies heavily on complementary in vitro and in vivo experimental approaches. In vitro methods employ cell-based assays to measure endpoints such as cytotoxicity, inflammatory responses, reactive oxygen species generation, and membrane or cellular damage following particle exposure under controlled laboratory conditions. In vivo studies use animal models to investigate systemic and organ-level toxicological effects, including pulmonary inflammation, fibrosis, and genotoxicity. However, conventional approaches can be resource-intensive, time-consuming, and ethically constrained, particularly when large numbers of materials with diverse physicochemical properties must be assessed. These limitations have motivated the development of high-throughput, mechanistic, and computational approaches capable of screening larger numbers of materials while reducing reliance on animal testing (Shatkin et al., 2013; Shatkin & Ong, 2016; Liu et al., 2021).

Quantitative Structure-Activity Relationship (QSAR) models represent the primary computational alternative, predicting toxicity from chemical structural descriptors using statistical machine learning (Shatkin et al., 2013; Shatkin & Ong, 2016). QSAR approaches have been widely applied to pharmaceutical compounds and simple nanoparticles, but the application to complex industrial materials exposes fundamental limitations (Liu et al., 2021; Puzyn et al., 2009). Standard Nano-QSAR models are trained on structurally uniform, symmetric nanoparticles such as gold nanospheres, silver nanocubes, and titanium dioxide particles, and rely on global descriptors such as total surface charge (zeta potential), particle diameter, and bulk chemical formula (Puzyn et al., 2009; Toropov & Toropova, 2015).

These descriptors fail when applied to irregular, multi-mineral particles such as cement clinker. The biologically relevant interactions occur at specific, localised surface sites, for example calcium coordination environments, silanol groups, aluminate bridges and iron coordination centres, and these vary dramatically across the material. A static chemical formula cannot capture these complex, spatially localised surface topologies. This literature identifies this as the "material heterogeneity problem" of Nano-QSAR: the predictive power of existing models breaks down precisely for the most practically important class of industrial particle, the multi-mineral composite.

## 1.3 The Physics-Based Solution

This thesis addresses the material heterogeneity problem by developing a physics-driven computational pipeline that replaces static structural descriptors with dynamic, spatially resolved free energy interaction profiles. Rather than asking _"what is the bulk composition of this particle?"_, the approach asks, _"how does the actual atomic-level surface of this material interact with each type of biological molecule in aqueous solution?"_ - a question that is both more meaningful and more directly relevant to toxicological results.

The central computational tool used here is the Potential of Mean Force (PMF), a free energy profile describing the thermodynamic interaction between a surface and a biomolecule as a function of their separation distance. The PMF captures the physically relevant features of a surface-molecule interaction: the strength of the primary binding event, the spatial range of the interaction, the energetic cost of displacing the hydration layer and the existence of secondary binding states. These features are directly connected to biological outcomes. Strong adsorption events can denature proteins, disrupt membrane integrity, and trigger inflammatory cascades (Lynch and Dawson, 2008; Nel et al., 2009).

Computing PMFs via classical molecular dynamics simulation requires days to weeks of computation per surface-molecule pair. The PMFPredictor machine learning toolkit (Rouse and Lobaskin, 2023) dramatically accelerates this by training a neural network on a library of atomistic MD-computed PMFs and using it to predict new PMFs from a compact physical representation. This enables informative screening of an entire library of biomolecules against multiple mineral surfaces in hours rather than years.

## 1.4 Project Objectives

The specific objectives of this project, as defined in collaboration with supervisor Professor Vladimir Lobaskin, are:

- To construct atomistic slab models of the four principal cement mineral phases (C3S, C2S, C3A, C4AF) from crystallographic reference structures, using CHARMM-GUI for phases with direct support and a custom CIF-conversion workflow for the remaining phases.
- To generate surface-molecule interaction potentials and PMF curves for each mineral phase using the PMFPredictor machine-learning toolkit for a library of 89 biomolecules.
- To implement Route 1: compute **per-phase** PMF predictions and combine them via a weighted average based on cement mineral volume fractions to produce a representative OPC interaction profile.
- To implement Route 2: construct a **spatially patched** composite surface representing the OPC mineral composition and run the PMFPredictor directly on this combined structure.
- To compare the results of Route 1 and Route 2, checking whether treating cement as a spatially heterogeneous surface produces different predictions than the weighted average of homogeneous phase results.
- To extend the Route 1 analysis to Fly Ash and Slag cement formulations, comparing biomolecule binding profiles across cement types and identifying composition - dependent differences in binding selectivity.

The central research questions addressed in this thesis are:

Do the four principal cement mineral phases show distinct and chemically explainable biomolecule adsorption profiles? Can these per-phase profiles be combined through a physically motivated weighting scheme to produce a representative interaction profile for multi-phase cement formulations? And does treating cement as a spatially heterogeneous composite surface produce different predictions compared to a linear combination of homogeneous phase results?

The central hypothesis is that the four principal cement mineral phases have chemically distinct biomolecule adsorption profiles reflecting their surface compositions. These profiles can be combined using experimentally determined mineral volume fractions to create a representative interaction fingerprint for multi-phase cement. A secondary hypothesis is that treating cement as a spatially heterogeneous surface, where molecules may simultaneously interact with sites from adjacent mineral phases, will produce systematically different predictions compared to the simple weighted average. Testing these hypotheses computationally, at a scale and speed not achievable by direct molecular dynamics simulation, is the core contribution of this thesis.

# Chapter 2: Background and Literature Review

## 2.1 Cement Composition and Mineral Phases

Ordinary Portland Cement (OPC) is produced by the clinkering of limestone (CaCO3) and clay minerals in a rotary kiln at approximately 1450 degrees Celsius (Taylor, 1997; Stutzman et al., 2015). The resulting clinker consists of four principal mineral phases whose relative proportions are determined by the raw material composition and kiln operating conditions (Taylor, 1997; Stutzman et al., 2015). Upon grinding and hydration, these phases react with water to produce the calcium silicate hydrate (C-S-H) gel responsible for cement's structural properties (Taylor, 1997). Understanding the surface chemistry of the anhydrous clinker phases is particularly relevant for occupational exposure scenarios, where freshly ground cement dust is inhaled before notable hydration has occurred (Pietsch et al., 2021).

The four principal mineral phases of OPC and their typical volume percentages are (Taylor, 1997):

- Tricalcium Silicate (Ca3SiO5, abbreviated C3S using cement chemistry notation, mineralogically known as Alite): 50-70% by volume. C3S is the dominant and most important phase, contributing the majority of OPC's early strength. It crystallises in a monoclinic structure, and its surface is characterised by alternating calcium and silicate layers. C3S undergoes rapid hydration upon contact with water, releasing calcium ions and forming C-S-H gel and portlandite (Ca (OH)2). In this work, the (001) surface plane was used, constructed using CHARMM-GUI Nanomaterial Modeler.
- Dicalcium Silicate (Ca2SiO4, C2S, Belite, mineralogically known as Larnite in the beta polymorph): 15-30% by volume. C2S is the second most abundant phase and contributes primarily to late-stage strength development (28 days and beyond). The beta polymorph (beta-C2S) is the phase stable in cement clinker, crystallising in space group P21/n. Its surface chemistry is like C3S but with a lower Ca:Si ratio (2:1 versus 3:1). The crystal structure used here was obtained from Crystallography Open Database entry 9012789 (Tsurumi et al., 1994).
- Tricalcium Aluminate (Ca3Al2O6, C3A): 5-10% by volume. C3A is the most reactive phase and is responsible for the initial flash setting of cement without gypsum retarder. It typically crystallises in the cubic structure used here- alkali substitution can produce other polymorphic forms, including orthorhombic and, at higher substitution levels, monoclinic forms. The cubic form is the phase present in ordinary Portland cement clinker. Its surface is dominated by aluminium coordination polyhedra rather than silicon, producing a distinct electrostatic environment. In CHARMM-GUI, only the (010) cleavage plane is available for C3A.
- Tetracalcium Aluminoferrite (Ca4Al2Fe2O10, C4AF, Brownmillerite): 5-15% by volume. C4AF is the interstitial phase that fills the spaces between C3S, C2S, and C3A in the clinker. It gives cement its characteristic grey colour through the optical absorption of Fe<sup>3+</sup> ions. Crystallographically, it features a disordered structure where Al<sup>3+</sup> and Fe<sup>3+</sup> partially occupy the same crystallographic positions. The theoretical stoichiometry Ca4Al2Fe2O10 implies a 1:1 Al:Fe ratio, but in practice the occupancies vary. The COD crystal structure used in this work (entry 1200009) (Colville and Geller, 1971) shows Al<sup>3+</sup> occupying approximately 76% and Fe<sup>3+</sup> approximately 24% of the mixed sites. This structural disorder is an important consideration when interpreting the surface chemistry predictions for this phase.

In addition to OPC, two supplementary cement formulations are considered in this work. Fly Ash cement incorporates coal combustion fly ash as a supplementary cementitious material. The fly ash phase composition differs substantially from OPC. It has a much higher C3A content (approximately 35%), reflecting the higher alumina content of coal ash, and a reduced C3S content (approximately 30%) (Bhatt et al., 2019). Slag cement (blast furnace slag cement) incorporates granulated blast furnace slag (Singh and Garg, 2006), which contributes higher C2S and C3A fractions relative to OPC. These compositional differences are expected to produce distinct biomolecule adsorption profiles. Fly ash and slag cements, however, are not simple four-phase clinker systems in the same sense as OPC, their composition depends on the source material and blending ratio with OPC clinker.

For the purposes of the Route 1 comparison in this thesis, representative phase fractions of C3S:0.30, C2S:0.20, C3A:0.35, C4AF:0.15 (Fly Ash) and C3S:0.40, C2S:0.30, C3A:0.20, C4AF:0.10 (Slag) were assigned as illustrative model compositions rather than as universally representative values for all fly ash or slag cement products.

## 2.2 Surface-Biomolecule Interactions at the Nano-Bio Interface

When a nanoparticle or fine particle enters a biological environment such as the lung lining fluid or cytoplasm, it rapidly acquires a coating of biological macromolecules - primarily proteins and lipids - forming a "protein corona" or "biomolecular corona". The composition of this corona is determined by the competitive adsorption of available biomolecules, governed by the free energy of each surface-molecule interaction. The corona changes the biological identity of the particle: cellular receptors do not "see" the particle surface but rather the protein corona coating it, meaning the corona composition determines which biological pathways are activated (Nel et al., 2009).

Strong adsorption events can lead to protein unfolding upon binding, exposing normally buried hydrophobic regions and potentially triggering inflammatory recognition by macrophages and dendritic cells (Nel et al., 2009; Kopac, 2021). Adsorbed proteins may also lose their native function - enzymes adsorbed to particles typically show reduced activity, and signalling proteins may be permanently deactivated (Lynch & Dawson, 2008; Nel et al., 2009). The strength of adsorption, characterised by the minimum free energy of the PMF curve (∆G<sub>min</sub>), is hence a direct predictor of the likelihood and severity of protein denaturation events (Rouse et al., 2021; Rouse & Lobaskin, 2023). Phospholipid adsorption is equally important: the pulmonary surfactant lining the alveoli consists primarily of phosphatidylcholine lipids, and adsorption of surfactant onto inhaled particles disrupts the surfactant monolayer, reducing its ability to lower surface tension and maintain alveolar patency (Liu et al., 2020). The affinity of different cement mineral phases for phospholipid headgroup analogues (DMEP, PHO) is therefore directly relevant to predicting pulmonary surfactant disruption (Rouse & Lobaskin, 2023).

## 2.3 The Potential of Mean Force: Theory and Interpretation

The Potential of Mean Force (PMF) W(z) is the reversible work required to move a particle from the bulk solution (z = infinity) to a distance z from the surface, while allowing all other degrees of freedom of the system (solvent molecules, ions, molecular orientations) to fully equilibrate at each step. It is formally defined by:

where kB is the Boltzmann constant, T is the absolute temperature, ρ(z) is the equilibrium probability density of finding the molecule at distance z from the surface, and ρ bulk is the corresponding bulk solution probability density. This definition sets W(z) = 0 in the bulk solution, so negative values of W(z) indicate attractive interactions and positive values indicate repulsion.

A typical PMF profile for a biomolecule approaching a polar inorganic surface through water has the following characteristic regions (Rouse et al., 2021):

- The steric repulsion wall (z < 0.2-0.3 nm): At short distances, the electron clouds of the surface and molecule overlap, producing a sharply rising repulsive potential. This defines the hard-core exclusion zone where the molecule cannot physically penetrate the surface.
- The primary adsorption minimum ∆G<sub>min</sub> (typically z = 0.3-0.5 nm): The global minimum of the PMF, representing the thermodynamically most stable bound state. The depth of this minimum is the adsorption free energy - the key thermodynamic descriptor of surface binding affinity. More negative values indicate stronger thermodynamic binding. Values below approximately -20 kJ/mol in this dataset indicate particularly strong predicted adsorption, but this should not be taken as proof of irreversible binding without kinetic or experimental validation.
- The desolvation barrier ∆G-barrier (typically z = 0.4-0.6 nm): The PMF desolvation barrier height, defined operationally as the difference between the primary adsorption minimum and the subsequent local maximum in the PMF curve, was computed for each surface-molecule pair. A high desolvation barrier height means the biomolecule must overcome significant energetic opposition to displace the surface hydration layer, even if the final adsorbed state is thermodynamically favourable.
- The secondary adsorption state (z = 0.5-0.8 nm): A shallow secondary minimum sometimes observed at intermediate distances, corresponding to a second-layer adsorption state where the molecule is separated from the surface by one or two water molecule layers.
- The bulk solution region (z > 0.8 nm): The PMF converges asymptotically to zero, indicating the surface has negligible influence on the molecule at these distances.

## 2.4 The PMFPredictor Machine Learning Toolkit

The PMFPredictor toolkit (Rouse and Lobaskin, 2023) predicts PMFs for arbitrary surface-molecule combinations without requiring full molecular dynamics simulation, using a physical fingerprinting approach that encodes both surface and molecule chemistry through their interactions with a set of chemical probe atoms.

For the surface, six probe types are used: point charge probes (representing electrostatic interactions), water (representing hydrogen bonding), waterUCD, methane (representing hydrophobic/van der Waals interactions), carbring (representing aromatic π-interactions), and cline3 (representing linear chain interactions). For each probe type, the free energy profile as a function of distance from the surface is computed analytically from the force field parameters of the surface atoms by Boltzmann averaging over a spatial grid of probe positions and orientations. This produces a set of six free energy curves that collectively fingerprint the electrostatic, hydrogen bonding, and dispersion interaction profile of the surface. These probe free energy curves are then compressed into a fixed-length vector using an expansion in hypergeometric basis functions. The hypergeometric expansion is particularly well-suited to PMF curves because its basis functions naturally capture the asymmetric shapes - steep repulsive walls and shallower attractive wells - characteristic of surface-molecule interactions. An analogous fingerprinting and expansion process is applied to the molecule.

A neural network ensemble is trained on a library of PMFs computed via full atomistic MD simulation, covering diverse inorganic surfaces and organic molecules. It then predicts the PMF for a new surface-molecule pair from the combined surface and molecular HGE coefficient vectors. The toolkit uses a bootstrapped ensemble of 10 models rather than a single model, providing both a mean prediction and an implicit measure of prediction uncertainty through ensemble variance. Predictions take seconds rather than the hours to days required for MD-based PMF computation.

The six probe types used to fingerprint each surface are designed to capture the dominant interaction mechanisms relevant to biomolecule binding. Point charge probes sample the electrostatic potential of the surface and capture Coulombic interactions with charged functional groups. The water probe represents hydrogen bonding capacity. Methane probes the hydrophobic and van der Waals character of the surface. The waterUCD probe is a modified water probe parameterised at University College Dublin that captures a broader range of polar interactions. The carbring probe represents aromatic ring interactions, capturing π-cation and π-π stacking contributions. The cline3 probe represents linear chain interactions relevant to aliphatic functional groups. Together these six probes provide a physically motivated and chemically interpretable fingerprint of the surface.

## 2.5 The INTERFACE Force Field

The INTERFACE force field (IFF, Heinz et al., 2013) provides validated Lennard-Jones and electrostatic parameters for modelling inorganic-organic interfaces in molecular dynamics simulations. IFF was parameterised to reproduce experimental measurements of lattice parameters, surface energies, interfacial energies and wetting contact angles for a wide range of inorganic materials including noble metals, metal oxides, clay minerals and calcium silicate phases. The critical feature for this application is that IFF uses the same functional form as the CHARMM36 and AMBER biomolecular force fields, enabling direct simulation of inorganic surfaces in contact with biological molecules without mixing incompatible force field styles.

The CHARMM-GUI Nanomaterial Modeler automates the construction and IFF parameterisation of atomistic slab models for a curated set of inorganic materials, outputting simulation-ready input files for GROMACS, NAMD, AMBER, and OpenMM. The atom type labels in the IFF GROMACS topology follow a systematic naming convention: the prefix I indicates INTERFACE FF atom type, followed by element and environment identifiers. For C3S, the four atom types are ICA_S (calcium in silicate environment, charge +1.5), ISC1 (silicon, charge +1.0), IOC1 (bridging oxygen, charge -1.0), and IOC2 (non-bridging oxygen, charge -1.5) (Heinz et al., 2013; Choi et al., 2022).

# Chapter 3: Computational Methods

## 3.1 Overview of the Computational Pipeline

The complete computational pipeline developed here consists of seven sequential stages. Stage 1 generates atomistic slab models for each cement mineral phase. Stage 2 converts these to the PMFPredictor CSV format. Stage 3 computes surface probe interaction potentials. Stage 4 generates hypergeometric expansion (HGE) coefficient vectors for all surfaces. Stage 5 generates HGE coefficients for the molecule library. Stage 6 runs the neural network prediction to produce PMF curves. Stage 7 extracts adsorption energies and performs the weighted combination analysis. All computational work was performed on the smartnano1.ucd.ie Ubuntu 22.04 LTS server running Python 3.10, with the PMFPredictor Toolkit cloned from GitHub (https://github.com/ijrouse/PMFPredictor-Toolkit) and pre-trained models obtained from Zenodo ([DOI: 10.5281/zenodo.7253566](https://zenodo.org/records/7253566)).

_Figure 3.1: Overview of the seven-stage computational pipeline described in Sections 3.2-3.8, from surface slab construction through to Route 1/Route 2 multi-phase combination._

## 3.2 Surface Slab Construction

### 3.2.1 C3S and C3A via CHARMM-GUI Nanomaterial Modeler

Tricalcium Silicate (C3S) and Tricalcium Aluminate (C3A) slabs were constructed using CHARMM-GUI Nanomaterial Modeler (Choi et al., 2022; Jo et al., 2008) accessed at https://charmm-gui.org. The Cement Minerals section of the Nanomaterial Modeler was used, which provides pre-parameterised crystal structures for these two phases with INTERFACE force field parameters. Both materials were made as box-shaped slabs with XY periodic boundary conditions (Periodic Options: X and Y ticked, Z unticked), representing infinite planar surfaces. Vacuum boundary conditions (System Type: Vacuum) were selected rather than explicit solvation, as the PMFPredictor toolkit encodes aqueous solvation effects implicitly through its training on explicitly solvated MD systems. GROMACS output format was selected for all files.

The C3S slab (CHARMM-GUI Job ID: 8472098836) was built with the (001) Miller index cleavage plane, producing an orthorhombic simulation box with dimensions 48.94 x 42.44 x 56.00 Å (A x B x C). The final slab contained 3,888 atoms with four distinct INTERFACE FF atom types:

- ICA_S (Ca, charge +1.500e, mass 40.08 AMU, sigma 0.2985 nm, epsilon 1.1715 kJ/mol)
- ISC1 (Si, charge +1.000e, mass 28.09 AMU, sigma 0.4009 nm, epsilon 1.9665 kJ/mol)
- IOC1 (O, charge -1.000e, mass 16.00 AMU, sigma 0.3118 nm, epsilon 0.3347 kJ/mol)
- IOC2 (O, charge -1.500e, mass 16.00 AMU, sigma 0.3118 nm, epsilon 0.3347 kJ/mol)

The two oxygen types represent bridging (IOC1) and non-bridging (IOC2) oxygen environments in the silicate network.

The C3A slab (CHARMM-GUI Job ID: 8553868909) was built with the (010) Miller index, the only cleavage plane available for this phase in CHARMM-GUI, reflecting the cubic crystal symmetry of C3A. The slab dimensions were 30.53 x 30.53 x 61.00 Å containing 2,112 atoms with atom types:

- IOC3, IOC4, IAC1 (Al, charge +1.200e, mass 26.98 AMU, sigma 0.4098 nm, epsilon 2.0920 kJ/mol)
- ICA_A (Ca in aluminate environment, charge +1.500e, mass 40.08 AMU, sigma 0.2940 nm, epsilon 0.6276 kJ/mol).

The CHARMM-GUI GROMACS output consists of: step3_input.gro (atomic coordinates), topol.top (topology with atom types and charges), and a toppar/ subdirectory containing force field parameters including forcefield.itp. These files were uploaded to the smartnano1 server and converted to PMFPredictor format using the toolkit's NPtoCSV.py script, which reads the GRO coordinate file, PSF topology, and ITP force field parameters and produces a unified CSV file containing AtomID, AtomName, AtomType, x\[nm\], y\[nm\], z\[nm\], charge\[e\], mass \[AMU\], sigma\[nm\], and epsilon\[kJ/mol\] for each atom.

### 3.2.2 C2S and C4AF via CIF File Conversion

Dicalcium Silicate (C2S) and Tetracalcium Aluminoferrite (C4AF) are not available in CHARMM-GUI Nanomaterial Modeler. Crystal structure data was obtained from the Crystallography Open Database (COD, http://www.crystallography.net/cod/ ). For C2S, COD entry [9012789](https://www.crystallography.net/cod/9012789.html) was selected, corresponding to the beta-polymorph (Larnite) characterised by Tsurumi et al. (1994) with lattice parameters a=5.5041 A, b=6.7622 A, c=9.3281 A, beta=94.172 degrees, space group P21/n (monoclinic, space group No. 14). This polymorph is the thermodynamically stable form of Ca2SiO4 at cement clinker temperatures and is the dominant C2S phase in Portland cement. For C4AF, COD entry [1200009](https://www.crystallography.net/cod/1200009.html) was selected, corresponding to the Brownmillerite structure characterised by Colville and Geller (1971) with space group Ibm2 (orthorhombic).

A custom Python script (cif_to_pmf_csv.py) was developed to convert these CIF files to the PMFPredictor format. The script uses the pymatgen Structure class (Ong et al., 2013) to parse the crystallographic data and build a supercell model. A 3x3x2 supercell expansion was applied to provide a slab of sufficient lateral dimensions for the surface potential calculation grid. This size was chosen to give lateral dimensions of at least 15 Å in both x and y, enough to avoid the probe molecule interacting with its own periodic images at the typical probe-surface separations used here (0.2 to 1.5 nm). This is consistent with the supercell sizes used for the pre-built surfaces in the PMFPredictor toolkit. INTERFACE force field parameters were assigned to each atom element based on the values extracted from the CHARMM-GUI-generated C3S and C3A GROMACS topologies, maintaining force field consistency across all four phases.

Two implementation challenges were addressed. First, pymatgen reads oxidation state information from CIF files and appends it to element symbols (e.g. Ca2+, O2-), requiring a parsing step to extract the base element symbol before parameter lookup. This was handled by a _get_element()_ function that strips oxidation state suffixes and calls _site.specie.symbol_ for ordered sites. Second, the C4AF Brownmillerite structure shows crystallographic site disorder: the 4a and 4b Wyckoff positions are occupied by both Al<sup>3+</sup> (76%) and Fe<sup>3+</sup> (24%) simultaneously. For the purposes of this work, each disordered site was assigned to the majority element (Al<sup>3+</sup> at 76% occupancy) using pymatgen's _species_string_ parsing. This approximation preserves the dominant chemical character of each site while acknowledging that the actual C4AF surface has a statistical mixture of Al and Fe coordination environments. This approximation carries a real consequence for the surface chemistry, though: the C4AF model used in this work is predominantly aluminate in character, with Al<sup>3+</sup> assigned to 76% of the mixed sites and Fe<sup>3+</sup> to only 24%. The Fe<sup>3+</sup> centres, which are expected to drive the strongest Lewis acid interactions with biological oxygen donor groups such as carboxylates and phosphates, are therefore underrepresented in the model relative to a real C4AF surface. This means the binding predictions for C4AF should be interpreted as a conservative estimate. A real C4AF surface with its true mixed Al/Fe chemistry would be expected to show equal or stronger binding than the predictions reported here. This limitation is discussed further in Section [6.4](#_6.4_Limitations_and).

The C2S 3x3x2 supercell produced a simulation box of approximately 16.51 x 20.29 x 18.66 Å, containing 504 atoms. The C4AF 3x3x2 supercell produced a box of approximately 16.63 x 30.77 x 11.09 Å, containing 648 atoms. Both slabs are significantly smaller than the CHARMM-GUI-derived C3S and C3A slabs, reflecting the smaller unit cells of the CIF-sourced structures.

## 3.3 Surface Potential Generation

Surface probe interaction potentials were made for all four cement mineral phases using _GenerateSurfacePotentials.py_. For each surface, the script calculates the free energy profile W<sub>probe(z)</sub> of six probe types as a function of distance z from the surface. The calculation proceeds by:

- Defining a regular grid of probe positions in the XY plane at each distance z from the surface (50 x 50 grid points\* across the unit cell).
- For each grid position, computing the total Lennard-Jones and electrostatic interaction energy between the probe and all surface atoms within a cutoff radius.
- Boltzmann-averaging across all grid positions to obtain the free energy at distance z:

where w<sub>i</sub> are statistical weights.

- Repeating this across all distances from z = 0.2 nm to z = 1.5 nm in steps of 0.001 nm.

\*The same 50 x 50 grid point density was applied to all surfaces regardless of unit cell size. For the larger CHARMM-GUI-derived slabs (C3S: 48.94 x 42.44 Å) this corresponds to a lateral grid spacing of approximately 0.98 x 0.85 Å per probe position. For the smaller CIF-derived slabs the grid spacing is proportionally finer, providing better spatial sampling of the smaller unit cell. This difference in effective resolution was assumed, consistent with the toolkit defaults, not to significantly affect the resulting probe potential profiles, though this was not independently verified through a resolution convergence test.

Surface potential files are saved for each probe type in the _SurfacePotentials/_ directory. The _ZeroLevelOffset_ and alignment offsets are computed by comparing the probe potential maximum location to a reference PMF curve (alanine on the surface) and saved in _Datasets/SurfaceOffsetData.csv_ for use by the _HGExpand_ stage.

The total computation time for surface potential generation across all four cement phases was approximately 13.5 hours on a single CPU core of the smartnano1.ucd.ie server (AMD/Intel architecture, Ubuntu 22.04). Individual phase runtimes were: C3S 7.5 hours (3,888 atoms), C3A 4.5 hours (2,112 atoms), C2S 40 minutes (504 atoms), and C4AF 54 minutes (648 atoms). The dramatically longer runtimes for the CHARMM-GUI-derived slabs compared to the CIF-derived slabs reflect the much larger supercell sizes - the CHARMM-GUI generates larger unit cells by design to represent the periodic surface adequately, while the 3x3x2 supercell of the CIF structures produces smaller but still representative models.

## 3.4 Hypergeometric Expansion of Surface and Chemical Potentials

The surface probe potentials W<sub>probe(z)</sub> were compressed into fixed-length descriptor vectors using the hypergeometric expansion implemented in _HGExpandSurfacePotential.py_. The expansion represents each probe free energy profile as a series:

where HGEFunc(z, r0, n) are hypergeometric basis functions centred at the surface exclusion radius r<sub>0</sub>, and A<sub>n</sub> are the expansion coefficients determined by numerical integration. An expansion order of N=20 was used, consistent with the default configuration of the PMFPredictor toolkit as established by Rouse and Lobaskin (2023). No independent convergence testing of the expansion order was performed here. The coefficients for all six probe types, combined with surface geometry parameters (r0, alignment offsets), constitute the surface descriptor vector.

Chemical HGE coefficients were calculated analogously using _HGExpandChemicalPotential.py_ for the 89 chemicals in the molecule library. Compatibility issues with SciPy v1.15+ (which removed the scipy.integrate.quadrature function) required patching the source code to use the equivalent scipy.integrate.quad function with equivalent tolerance settings. Additionally, a version incompatibility between the pre-trained model checkpoints (saved in TensorFlow 2.10 SavedModel format) and the installed TensorFlow 2.21 (which uses Keras 3 with an incompatible SavedModel loading API) required downgrading to TensorFlow 2.10.0 with NumPy 1.26.4.

## 3.5 PMF Prediction using Neural Network Ensemble

PMF curves were predicted using _BuildPredictedPMFs.py_ with a bootstrapped ensemble of 10 pre-trained neural network models (PMFPredictor-oct13-simplesplit-bootstrapped-ensemble1 through ensemble10), obtained from Zenodo ([DOI: 10.5281/zenodo.7253566](https://zenodo.org/records/7253566)). The script reads the surface and chemical HGE coefficient vectors from _Datasets/SurfacePotentialCoefficients-oct12.csv_ and the analogous chemical dataset, combines them into a single input feature matrix, and predicts the HGE output coefficients for each surface-molecule pair using each ensemble model. The predicted HGE coefficients are converted back to PMF curves by evaluating the hypergeometric series at 1300 distance points from r0 to 1.5 nm, normalized so that W(1.5 nm) = 0. Each of the 10 ensemble models produces an independent PMF prediction- the ensemble mean is used as the primary result, while the ensemble spread provides implicit uncertainty quantification. PMF files are saved for each surface-molecule pair.

## 3.6 Adsorption Energy Extraction

Adsorption energies (∆G<sub>min</sub> values) were extracted from the predicted PMF curves as the global minimum value over the range z = 0.2 to 1.5 nm. The binding distance (r<sub>min</sub>) was recorded as the z-value at which the minimum occurs. Desolvation barrier heights (∆G-barrier) were computed as the difference between the maximum PMF value in the range beyond the primary minimum (z > r<sub>min</sub>) and the minimum value. These three parameters - ∆G<sub>min</sub>, r<sub>min</sub>, and ∆G barrier - constitute the primary quantitative outputs of the pipeline for each surface-molecule pair.

## 3.7 Route 1: Weighted Average Combination

The first approach for generating a representative cement PMF combines the per-phase predictions using a linear weighted average based on the volume fractions of each mineral phase. For a given biomolecule, the composite PMF is computed as:

where the weights sum to unity and represent the volume fractions of each phase. For OPC: w<sub>C3S</sub> = 0.65, w<sub>C2S</sub> = 0.15, w<sub>C3A</sub> = 0.10, w<sub>C4AF</sub> = 0.10. For Fly Ash cement: w<sub>C3S</sub> = 0.30, w<sub>C2S</sub> = 0.20, w<sub>C3A</sub> = 0.35, w<sub>C4AF</sub> = 0.15. For Slag cement: w<sub>C3S</sub> = 0.40, w<sub>C2S</sub> = 0.30, w<sub>C3A</sub> = 0.20, w<sub>C4AF</sub> = 0.10. This calculation was implemented in the custom script _weighted_cement_pmf.py_, applied to all 89 molecules in the library for each cement type.

Mineral volume fractions are used here as a proxy for the relative contribution of each phase to the composite interaction profile, not as a direct measurement of exposed surface area. Real cement particles have complex morphology in which surface area exposure of each phase may differ from its bulk volume fraction. Route 1 should therefore be understood as a composition-weighted approximation rather than a physically exact model of the particle surface.

Two implementations of this calculation were used in this thesis. A single-model script ([Appendix B.2](#_B.2a_Single-model_Route)) computes the weighted average using predictions from ensemble model 1 only; this was used to generate the cement type comparison in [Section 5.2](#_5.2_Cement_Type) (Table 5.2, Figures 5.1 and 5.2). A separate ensemble-averaging implementation instead computes the adsorption energy independently for each of the 10 bootstrap models by taking the minimum of that model's weighted PMF curve, then reports the mean and standard deviation across these 10 independent values; this was used to generate the OPC results in [Section 5.1](#_5.1_Route_1:) (Table 5.1) and the per-phase ensemble tables in [Section 4.6](#_4.6_Ensemble-Averaged_Adsorption) (Tables 4.1 to 4.3). Both scripts are provided in Appendix B for reproducibility.

## 3.8 Route 2: Spatially Patched Composite Surface

The second approach for generating a representative cement interaction profile constructs a single composite surface that spatially tiles patches of each mineral phase in proportions matching the cement volume fractions, rather than combining per-phase PMF predictions after the fact. This was implemented in the custom script _patched_surface.py_ ([Appendix B.3](#_B.3_patched_surface.py)), which reads the atomic coordinate and force field CSV files for each cement mineral phase and concatenates subsets of each in the correct proportions.

An initial implementation targeted a composite of 2,016 atoms using full-depth mineral slabs taken directly from the individual phase CSVs (1,310 atoms C3S, 302 atoms C2S, 202 atoms C3A, 202 atoms C4AF), with patches placed side by side along the x-axis separated by 0.5 nm gaps. Surface potential generation on this structure (_GenerateSurfacePotentials.py_) completed without error, but the resulting probe free energy profiles failed to converge at large probe-surface separation, showing energies of approximately -100 kJ/mol at 1.5 nm where single-phase surfaces converge to approximately -4 to -7 kJ/mol. This was traced to the large z-extent of the combined slab, up to 5.1 nm across all four stacked phases, which exceeds the depth for which the surface alignment and hypergeometric expansion procedures in the PMFPredictor toolkit are designed. The thin lateral patches, taken from full slab depth without adjustment, also lacked consistent z-coordinate reference frames across phases, since the C2S and C4AF CIF derived slabs were not originally normalised to the same z convention as the CHARMM-GUI derived C3S and C3A slabs.

The implementation was revised to address both issues. Each phase was first independently normalised so that its uppermost atom sits at z = 0, then only the top 0.5 nm surface layer of each phase was retained, discarding the underlying bulk atoms. These thin surface layers were then linked with the same 0.5 nm x-direction gaps as before. This produced the final composite surface, PatchedOPCcement, containing 576 atoms distributed as 375 atoms C3S (65%), 87 atoms C2S (15%), 57 atoms C3A (10%), and 57 atoms C4AF (10%), with a total z-extent of 0.5 nm. This is the structure used for all Route 2 results reported in [Section 5.3](#_5.3_Route_2:) and corresponds to the script version listed in [Appendix B.3](#_B.3_patched_surface.py).

The composite surface was registered in _SurfaceDefinitions.csv_ and submitted to the full PMFPredictor pipeline. Surface potential generation completed successfully for all six probe types on this revised structure. The outcome of the later processing steps, and the reasons the pipeline could not be carried through to final PMF predictions, are discussed in [Section 5.3](#_5.3_Route_2:).

# Chapter 4: Results - Individual Phase PMF Predictions

## 4.1 Overview and Qualitative Validation

The results in this chapter are presented from two perspectives. Sections 4.2 to 4.5 discuss PMF curve shapes and individual binding features using values from a single representative model (ensemble model 1). These values are used to describe the qualitative character of each surface. Quantitative comparisons presented as ensemble results, including the adsorption energy tables in [Section 4.6](#_4.6_Ensemble-Averaged_Adsorption) and the OPC analysis in [Section 5.1](#_5.1_Route_1:), use the mean across all 10 bootstrap models, reported as mean plus or minus one standard deviation. The cement type comparison in [Section 5.2](#_5.2_Cement_Type) uses ensemble model 1 and is explicitly identified as a single-model analysis. Where a specific value is quoted in the text, single-model values are followed by (single model) and ensemble values are followed by (ensemble mean).

PMF curves were successfully predicted for all four cement mineral phases interacting with 89 biomolecules using the PMFPredictor ensemble. All predicted curves show the physically expected qualitative features - strong repulsive walls at short range, primary adsorption minima between 0.3-0.5 nm, desolvation barriers and convergence to zero at large separation. This is consistent with PMFs computed via full atomistic MD simulation for inorganic surfaces in the PMFPredictor training set, including gold, silver, and aluminium surfaces (Rouse and Lobaskin, 2023), and with PMFPredictor predictions for aluminium surfaces used in corona formation studies (Power et al., 2024). The qualitative agreement with physical expectations provides confidence that the ML model is producing chemically reasonable predictions, even for cement phases outside its original training distribution.

Figure 4.1 shows the predicted PMF curves for glycine - the simplest amino acid and a standard reference molecule in surface interaction studies - on all four cement mineral phases. The curves clearly differentiate the four phases and span a range of adsorption energies from approximately -2 kJ/mol (C3A) to -11 kJ/mol (C4AF), confirming that the model captures meaningful differences in surface chemistry across the cement phases.

_Figure 4.1: Predicted PMF curves for glycine (GLY) on all four cement mineral phases. All curves show the expected PMF shape. C4AF shows the deepest adsorption minimum while C3A shows the shallowest, reflecting differences in surface electrostatic environments across the four phases._

## 4.2 Tricalcium Silicate (C3S) - Results and Analysis

Note: Values in this section come from a single ensemble member (model 1), chosen to keep the qualitative discussion below simple; the full ten-model ensemble average, with uncertainty, is given in Section 4.6.

The C3S (001) surface shows high binding affinity across the amino acid library, with adsorption energies ranging from -0.6 kJ/mol (alanine, the smallest nonpolar residue) to -23.4 kJ/mol (tryptophan, the largest aromatic residue). The glycine adsorption minimum of -9 kJ/mol at approximately 0.35 nm is consistent with the binding affinities reported for calcium silicate surfaces in the PMFPredictor literature for similar molecules.

_Figure 4.2: Predicted PMF curve for glycine on the Tricalcium Silicate (C3S, Alite) (001) surface. Adsorption minimum of -9 kJ/mol at ~0.35 nm, desolvation barrier height 7.88 kJ/mol._

The C3S surface shows particularly stable binding for aromatic amino acids (tryptophan -23.4 kJ/mol (single model), tyrosine -15.9 kJ/mol, phenylalanine -8.4 kJ/mol) and charged residues (arginine -18.3 kJ/mol, histidine -16.8 to -18.3 kJ/mol). This pattern reflects the calcium-rich surface chemistry of C3S: aromatic rings interact with Ca<sup>2+</sup> ions through π-cation interactions, a well-established non-covalent interaction in which the diffuse negative charge of the aromatic π-system coordinates with the divalent calcium cation (Dougherty, 1996). The large desolvation barrier of 7.88 kJ/mol indicates a well-organised surface hydration layer that must be partly disrupted for adsorption to occur.

## 4.3 Tricalcium Aluminate (C3A) - Results and Analysis

Note: As in Section 4.2, the numbers quoted here are single-model (ensemble model 1) values, used for qualitative discussion only. Section 4.6 gives the ensemble-averaged equivalents with uncertainty.

The C3A (010) surface shows the weakest binding affinity among the four cement phases for most amino acids, with adsorption energies ranging from -0.1 kJ/mol (alanine, cysteine, serine) to -10.9 kJ/mol (tryptophan). The glycine adsorption energy of -2.0 kJ/mol is notably shallow, and many small amino acids show almost no affinity for the C3A surface.

_Figure 4.3: Predicted PMF curve for glycine on the C3A (010) surface. The shallow adsorption minimum of approximately -2 kJ/mol and low desolvation barrier (1.89 kJ/mol) indicate weak surface-biomolecule interactions._

The relatively low binding affinity of C3A for amino acids is chemically interpretable. The C3A surface is characterised by Al<sup>3+</sup> coordination polyhedra - predominantly AlO4 and AlO6 units - without the Fe<sup>3+</sup> centres or high Ca<sup>2+</sup> density that drive strong binding in C4AF and C3S respectively. The low desolvation barrier (1.89 kJ/mol) suggests that water is only weakly organised at the C3A surface, consistent with the more open aluminate framework structure. Notably, C3A shows relatively stronger affinity for phosphate-containing molecules (DMEP: -16.4 kJ/mol, PHO: -16.0 kJ/mol), revealing that aluminate surfaces interact selectively with phosphate groups through Al<sup>3+</sup>\-phosphate coordination, even while showing weak affinity for most amino acids.

## 4.4 Dicalcium Silicate (C2S) - Results and Analysis

Note: Single-model (ensemble model 1) values are used throughout this section for consistency with the other phase discussions; see Section 4.6 for the ensemble-averaged values and their uncertainty.

The C2S surface, constructed from COD entry [9012789](https://www.crystallography.net/cod/9012789.html), shows intermediate binding affinity with adsorption energies ranging from -0.1 kJ/mol to -17.6 kJ/mol. The glycine adsorption energy of -5.0 kJ/mol is intermediate between C3S and C3A, reflecting the intermediate Ca:Si ratio (2:1) of C2S compared to C3S (3:1). The desolvation barrier of 3.99 kJ/mol is modest.

_Figure 4.4: Predicted PMF curve for glycine on the C2S (Belite/Larnite) surface. Structure from COD 9012789. Adsorption minimum of -5.0 kJ/mol, desolvation barrier 3.99 kJ/mol._

The C2S binding pattern is generally similar to C3S but weaker across all amino acid classes, consistent with the lower surface calcium density. Phosphate analogues show strong affinity (PHO: -17.6 kJ/mol, DMEP: -15.2 kJ/mol), like C3A, reflecting the contribution of silicate surface groups to phosphate binding through Si-OH hydrogen bonding in addition to Ca<sup>2+</sup>\-phosphate electrostatic interactions.

## 4.5 Tetracalcium Aluminoferrite (C4AF) - Results and Analysis

Note: This section again uses single-model (ensemble model 1) values to describe qualitative trends; ensemble-averaged values with uncertainty are given in Section 4.6.

The C4AF (Brownmillerite) surface shows the strongest overall binding affinity, with adsorption energies ranging from -0.6 kJ/mol (alanine) to -34.4 kJ/mol (PHO-AC, phosphate analogue). These are the most negative adsorption energies in the entire dataset - PHO and DMEP energies below -30 kJ/mol on C4AF indicate particularly strong predicted adsorption but should not be interpreted as proof of irreversible binding without kinetic or experimental validation.

_Figure 4.5: Predicted PMF curve for glycine on the C4AF (Brownmillerite) surface. COD_ [_1200009_](https://www.crystallography.net/cod/1200009.html)_. Adsorption minimum -11 kJ/mol with the highest desolvation barrier (11.05 kJ/mol) among all four phases, indicating a strongly structured surface hydration layer._

The strong C4AF binding is consistent with contributions from the minority Fe<sup>3+</sup> sites together with the broader electrostatic character of the Brownmillerite surface. The exceptional glutamate affinity on C4AF (-21.6 kJ/mol compared to -9.6 kJ/mol on C3S) is consistent with coordination between the glutamate carboxylate group and Fe³⁺ surface sites, though this calculation cannot confirm that mechanism directly. As discussed in [Section 6.1](#_6.1_Physical_Interpretation), the element-level force field representation used for C4AF does not permit decomposition of the total adsorption energy into separate Fe and Al contributions.

## 4.6 Ensemble-Averaged Adsorption Energies

All ensemble-averaged PMF predictions reported in this thesis use the 10-model bootstrapped ensemble (PMFPredictor-oct13-simplesplit-bootstrapped-ensemble1 through ensemble10). For each surface-molecule pair, the adsorption energy was computed independently for each of the 10 models by taking the minimum of that model's predicted PMF curve. The reported mean and standard deviation are calculated across these 10 independent adsorption energy values. Adsorption energies are reported as mean ± 1 standard deviation in kJ/mol throughout. Figure 4.6 shows the ensemble-averaged PMF curves for glycine on all four cement phases, with shaded bands representing ±1 standard deviation across the 10 models. The uncertainty is largest in the repulsive wall region (z &lt; 0.35 nm), where individual model predictions diverge most significantly, and smallest in the bulk solution region (z &gt; 0.8 nm) where all models correctly predict near-zero interaction energy.

_Figure 4.6: Ensemble-averaged PMF curves for glycine on all four cement phases. Solid lines show the mean prediction across 10 bootstrap models; shaded regions show the pointwise ±1 standard deviation of the PMF values across models. Adsorption-energy means and standard deviations were calculated separately from the model-specific PMF minima. C4AF shows the deepest mean adsorption energy (-9.82 ± 2.88 kJ/mol), closely followed by C3S (-9.26 ± 1.93 kJ/mol); given the overlapping uncertainty ranges, these two phases are not distinguishable for glycine binding. C3A (-3.08 ± 1.99 kJ/mol) and C2S (-2.55 ± 2.08 kJ/mol) show weaker binding, with overlapping uncertainty ranges._

The ensemble-averaged adsorption energies for glycine differ from the single-model results in an important respect: while the single model suggested a clear four-way ranking (C4AF > C3S > C2S > C3A), the corrected ensemble calculation shows that C3S and C4AF cannot be clearly distinguished within the model uncertainty, as can C3A and C2S. The finding that the Ca-rich C3S and mixed Al/Fe C4AF phases bind glycine more strongly than the aluminate-dominated phases (C3A, C2S) is robust across the full ensemble, but a clean ranking among all four individual phases is not supported once model-to-model uncertainty is properly propagated.

|     |     |     |     |
| --- | --- | --- | --- |
| **Phase** | **GLY EAds mean (kJ/mol)** | **GLY EAds std (kJ/mol)** | **Single model value (kJ/mol)** |
| C3S | \-9.26 | ±1.93 | \-9.0 |
| C3A | \-3.08 | ±1.99 | \-2.0 |
| C2S | \-2.55 | ±2.08 | \-5.0 |
| C4AF | \-9.82 | ±2.88 | \-11.0 |

_Table 4.1: Comparison of ensemble-averaged and single-model adsorption energies for glycine on all four cement phases. Ensemble averaging reduces apparent differences between phases while preserving the broader distinction between stronger and weaker binding groups._

The C2S ensemble value shows the largest divergence from its single-model prediction among the four phases, illustrating why single-model results should not be relied upon in isolation for quantitative comparison.

## 4.7 Tryptophan as a Reference Molecule

Among the 20 amino acid side chain analogues, tryptophan (TRP) consistently ranks as the strongest binder on C3S, C3A, and C4AF in the corrected ensemble predictions. On C2S, glutamate (GLUSCA-AC, -13.87 kJ/mol) narrowly exceeds tryptophan as the single strongest binder overall, a distinctive feature of this phase discussed further below. When phosphate analogues and sugars are included, DMEP and PHO overtake TRP on all three of the remaining phases. On C3S this is -20.68 versus -17.79 kJ/mol. On C3A it is -14.74 versus approximately -12.4 kJ/mol, using GLUSCA-JS as the strongest available amino acid. On C2S, DMEP and PHO reach -13.62 kJ/mol, well ahead of TRP. On C4AF, PHO-AC (-29.50 kJ/mol) far exceeds TRP. The following analysis focuses on tryptophan as a representative amino acid binder whose binding profile illustrates the π-cation interaction mechanism that distinguishes aromatic residues from aliphatic ones. Figure 4.7 shows the TRP PMF curves on all four cement phases, revealing the qualitative differences in binding profiles.

_Figure 4.7: Predicted PMF curves for tryptophan (TRPSCA-JS) on all four cement phases. TRP shows the deepest minimum on C3S (-23.4 kJ/mol), deeper than for glycine (-9 kJ/mol), reflecting the additional contribution of the indole ring to binding._

Comparing Fig 4.7 with the glycine PMF curves (Fig 4.1) reveals an important pattern: while glycine shows the ranking C4AF > C3S > C2S > C3A, tryptophan shows C3S > C4AF > C3A ≈ C2S (-10.9 vs -10.5 kJ/mol, too close at the single-model level to rank reliably), with C3S overtaking C4AF as the strongest binder. This inversion reflects the exceptional role of π-cation interactions between the tryptophan indole ring and Ca<sup>2+</sup> ions at the C3S surface - an interaction mechanism that is uniquely favoured by the high surface calcium density of C3S and is not replicated for glutamate (GLU) or other non-aromatic molecules.

_Figure 4.8: Side-by-side comparison of PMF curves for glycine (weakest OPC binder among amino acids) and tryptophan (strongest OPC binder) on all four cement phases. The dramatically deeper minima for tryptophan illustrate the importance of molecular functional group chemistry in determining surface adsorption affinity._

## 4.8 Desolvation Barriers

The desolvation barrier height is an important kinetic and thermodynamic parameter complementary to the adsorption energy, defined operationally as the difference between the primary adsorption minimum and the local maximum in the PMF curve. This quantity was computed for each surface-molecule pair. A high barrier means that even if the final adsorbed state is thermodynamically favourable, the molecule must overcome a significant energetic penalty to displace the surface hydration layer and reach the adsorption minimum. Figure 4.9 shows the desolvation barrier heights for glycine on all four cement phases.

_Figure 4.9: Desolvation barrier heights for glycine on the four cement mineral phases. C4AF shows the highest barrier (11.05 kJ/mol) and C3A the lowest (1.89 kJ/mol), correlating inversely with surface hydration layer organisation strength._

The pattern - C4AF (11.05) >> C3S (7.88) > C2S (3.99) > C3A (1.89) kJ/mol - is physically interpretable. C4AF, with its Fe<sup>3+</sup> centres, coordinates water strongly through iron-oxygen bonds, producing a tightly organised, difficult to displace hydration layer. C3S similarly coordinates water through Ca<sup>2+</sup>\-oxygen interactions. C3A, with the weakest overall binding, also shows the weakest water structuring, suggesting its surface is more dynamic and water exchange at the C3A surface is rapid. The correlation between high desolvation barrier height and high adsorption energy for C4AF suggests that the thermodynamic driving force for adsorption is sufficient to overcome even a significant kinetic barrier.

## 4.9 Top Binders Per Phase

Figure 4.10 shows the top 10 strongest binders for each individual cement phase, revealing the distinct binding selectivity profiles of each surface.

_Figure 4.10: Top 10 strongest binders for each cement mineral phase. Note the different x-axis scales - C4AF (bottom right) shows dramatically stronger binding for phosphate analogues (PHO, DMEP) than the other phases. C3A (top right) shows the narrowest range of adsorption energies, confirming its relatively low reactivity._

The most striking feature of Fig 4.10 is the exceptional phosphate binding on C4AF: PHO-AC (-34.4 kJ/mol) and DMEP-AC (-34.0 kJ/mol) on C4AF are more than twice as negative as their equivalents on C3S (~-15 to -16 kJ/mol). The presence of Fe<sup>3+</sup> sites provides a chemically plausible contribution to the strong phosphate affinity predicted for the C4AF surface.

The ensemble-averaged top 10 binders for each cement phase are presented below. Comparison with single-model results reveals that phosphate-containing molecules (DMEP, PHO) rank higher in the ensemble average than in the single-model predictions, while tryptophan ranks lower. This suggests the single model overestimates the contribution of π-cation interactions for aromatic residues on cement surfaces.

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **Rank** | **C3S** | **Energy ± std** | **C3A** | **Energy ± std** |
| 1   | DMEP-JS | \-20.68 ± 4.60 | DMEP-JS | \-14.74 ± 3.93 |
| 2   | PHO-JS | \-20.68 ± 4.60 | PHO-JS | \-14.74 ± 3.93 |
| 3   | DGL-AC | \-20.08 ± 4.91 | PHO-AC | \-13.50 ± 4.61 |
| 4   | BGLCNA-JS | \-19.00 ± 4.13 | DMEP-AC | \-13.48 ± 5.09 |
| 5   | ARGSCA-JS | \-18.31 ± 2.54 | GLUSCA-AC | \-13.45 ± 3.76 |
| 6   | TRPSCA-JS | \-17.79 ± 2.74 | GLUSCA-JS | \-12.43 ± 3.99 |
| 7   | BGLC-JS | \-16.85 ± 2.71 | BGLCNA-JS | \-11.89 ± 5.28 |
| 8   | DGL-JS | \-16.85 ± 2.71 | DGL-AC | \-11.57 ± 2.70 |
| 9   | HIPSCA-AC | \-16.58 ± 2.53 | CYMSCA-JS | \-10.92 ± 3.98 |
| 10  | HSPSCA-AC | \-16.58 ± 2.53 | AMAN-JS | \-9.74 ± 3.69 |

_Table 4.2: Top 10 binders for C3S and C3A (ensemble average ± std, kJ/mol)._

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **Rank** | **C2S** | **Energy ± std** | **C4AF** | **Energy ± std** |
| 1   | GLUSCA-AC | \-13.87 ± 5.41 | PHO-AC | \-29.50 ± 6.50 |
| 2   | DMEP-JS | \-13.62 ± 3.12 | DMEP-AC | \-28.53 ± 5.21 |
| 3   | PHO-JS | \-13.62 ± 3.12 | DMEP-JS | \-25.06 ± 4.32 |
| 4   | DGL-AC | \-11.40 ± 3.14 | PHO-JS | \-25.06 ± 4.32 |
| 5   | DMEP-AC | \-11.39 ± 4.82 | GLUSCA-AC | \-23.98 ± 1.83 |
| 6   | GLUSCA-JS | \-11.11 ± 4.37 | GLUSCA-JS | \-23.36 ± 4.89 |
| 7   | PHO-AC | \-10.93 ± 3.91 | ASPSCA-AC | \-22.50 ± 9.92 |
| 8   | CYMSCA-JS | \-10.81 ± 5.20 | ASPSCA-JS | \-22.21 ± 9.86 |
| 9   | ASPSCA-AC | \-10.30 ± 5.61 | CYMSCA-AC | \-19.07 ± 8.03 |
| 10  | BGLCNA-JS | \-9.14 ± 4.18 | CYMSCA-JS | \-18.70 ± 5.93 |

_Table 4.3: Top 10 binders for C2S and C4AF (ensemble average ± std, kJ/mol), computed as mean and standard deviation of adsorption energy calculated independently for each of 10 bootstrap models. Unlike the other three phases, C2S shows glutamate (GLUSCA-AC) rather than a phosphate analogue as its single strongest binder, a distinctive feature discussed further in Section 4.7. C4AF shows dramatically stronger phosphate binding than any other phase, with PHO-AC reaching -29.50 kJ/mol. The large uncertainty on ASPSCA (approximately ±10 kJ/mol) indicates high model-to-model variability for aspartate binding on C4AF._

### Full Amino Acid Library: All-Phase Comparison

To give a single consolidated view of amino acid selectivity across the four cement phases (for direct comparison rather than the per-phase presentation of Sections 4.2-4.5), Figure 4.11 plots the predicted adsorption energy of all 20 canonical amino acid side-chain analogues on each phase, ranked from strongest to weakest binder. Table 4.4 presents the same data in tabular form. Both use single-model (ensemble model 1) values, consistent with the per-phase discussion in Sections 4.2-4.5; the corresponding ensemble-averaged values for the top 10 binders per phase are given in Tables 4.2 and 4.3.

_Figure 4.11: Predicted adsorption energies of all 20 canonical amino acid side-chain analogues on the four cement mineral phases, ranked within each panel from strongest (top) to weakest (bottom) binder. Single-model (ensemble model 1) values._

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **Amino acid** | **C3S (kJ/mol)** | **C3A (kJ/mol)** | **C2S (kJ/mol)** | **C4AF (kJ/mol)** |
| ALA | \-0.6 | \-0.1 | \-0.1 | \-0.6 |
| ARG | \-18.3 | \-8.4 | \-7.2 | \-10.0 |
| ASN | \-5.9 | \-1.8 | \-2.7 | \-4.7 |
| ASP | \-5.4 | \-6.5 | \-5.2 | \-15.1 |
| CYS | \-4.4 | \-0.1 | \-0.1 | \-3.6 |
| GLN | \-10.4 | \-6.5 | \-5.8 | \-10.7 |
| GLU | \-9.2 | \-14.4 | \-9.3 | \-21.6 |
| GLY | \-9.0 | \-1.9 | \-5.0 | \-11.0 |
| HIS | \-11.1 | \-5.5 | \-6.0 | \-11.2 |
| ILE | \-5.1 | \-4.7 | \-3.3 | \-5.5 |
| LEU | \-5.9 | \-2.0 | \-4.4 | \-4.5 |
| LYS | \-11.3 | \-3.4 | \-6.6 | \-6.0 |
| MET | \-9.2 | \-5.2 | \-4.4 | \-12.3 |
| PHE | \-8.4 | \-1.6 | \-3.4 | \-11.7 |
| PRO | \-4.0 | \-1.3 | \-0.9 | \-2.2 |
| SER | \-2.8 | \-0.2 | \-0.7 | \-1.6 |
| THR | \-5.6 | \-2.1 | \-2.2 | \-3.4 |
| TRP | \-23.4 | \-10.9 | \-10.5 | \-19.9 |
| TYR | \-15.9 | \-8.2 | \-4.1 | \-15.1 |
| VAL | \-3.0 | \-0.7 | \-0.2 | \-2.2 |

_Table 4.4: Predicted adsorption energies (kJ/mol) for all 20 canonical amino acid side-chain analogues on the four cement mineral phases (single-model, ensemble model 1). Data are the amino-acid subset of Appendix Table A.1._

# Chapter 5: Results - Multi-Phase Combination and Cement Type Comparison

## 5.1 Route 1: Weighted Average PMF for OPC

This chapter presents the results of combining the per-phase PMF predictions from Chapter 4 into representative interaction profiles for complete cement formulations using both Route 1 (weighted average) and Route 2 (patched surface). Quantitative comparisons presented as ensemble results, including the adsorption energy tables in [Section 4.6](#_4.6_Ensemble-Averaged_Adsorption) and the OPC analysis in this section, use the mean and standard deviation across all 10 bootstrap models. The cement type comparison across OPC, Fly Ash, and Slag formulations in [Section 5.2](#_5.2_Cement_Type) uses ensemble model 1 only and is explicitly identified there as a single-model analysis. Applying Route 1 to OPC with weights C3S:0.65, C2S:0.15, C3A:0.10, C4AF:0.10, the ensemble-averaged top 10 binders are presented in Table 5.1. Phosphate and sugar analogues now rank above tryptophan, in contrast to the single-model result where tryptophan was the top binder. This shift reflects the ensemble consensus that phosphate binding is more consistently predicted across models than aromatic amino acid binding.

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **Rank** | **Molecule** | **Chemical Class** | **EAds mean (kJ/mol)** | **EAds std (kJ/mol)** |
| 1   | DMEP-JS | Dimethyl phosphate | \-19.25 | ±3.77 |
| 2   | PHO-JS | Phosphate analogue | \-19.25 | ±3.77 |
| 3   | DGL-AC | Galactose | \-17.31 | ±3.45 |
| 4   | BGLCNA-JS | N-acetylglucosamine | \-15.90 | ±3.55 |
| 5   | TRPSCA-JS | Tryptophan side chain | \-14.38 | ±2.38 |
| 6   | DMEP-AC | Dimethyl phosphate | \-14.35 | ±3.53 |
| 7   | ARGSCA-JS | Arginine side chain | \-14.19 | ±1.80 |
| 8   | GLUSCA-AC | Glutamate side chain | \-14.13 | ±4.21 |
| 9   | BGLC-JS | Glucose analogue | \-13.95 | ±2.06 |
| 10  | DGL-JS | Galactose | \-13.95 | ±2.06 |

_Table 5.1: Top 10 binders for weighted OPC cement, computed as mean and standard deviation of adsorption energy calculated independently for each of 10 bootstrap models. Phosphate analogues rank above tryptophan compared to single-model predictions. Arginine remains a top binder with relatively low uncertainty (±1.80 kJ/mol), suggesting this prediction is robust across ensemble models._

The shift in OPC top binder from tryptophan (single model: -19.65 kJ/mol) to DMEP/PHO (ensemble: -19.25 kJ/mol) is important for the toxicological interpretation. Both single-model and ensemble results agree that phospholipid headgroup analogues and aromatic amino acids are among the strongest binders. However, the ensemble results are more confident in the phosphate predictions than the aromatic amino acid predictions, since the phosphate predictions have lower relative uncertainty. This is consistent with the physical expectation that electrostatic Ca<sup>2+</sup>\-phosphate interactions are more straightforwardly captured by the PMFPredictor feature representation than the more complex π-cation interactions governing aromatic binding.

The arginine prediction (ARGSCA-JS: -14.19 ± 1.80 kJ/mol) has the smallest relative uncertainty among the top 10 binders, suggesting this is the most robustly predicted strong binder. Arginine interacts with cement surfaces through its guanidinium group, which carries a delocalised positive charge and engages in multiple simultaneous hydrogen bonds with surface oxygen atoms - a well-defined, electrostatic interaction that the PMFPredictor feature set is well-suited to capture.

## 5.2 Cement Type Comparison

The Route 1 analysis was extended to Fly Ash cement (C3S:0.30, C2S:0.20, C3A:0.35, C4AF:0.15) and Slag cement (C3S:0.40, C2S:0.30, C3A:0.20, C4AF:0.10). Figure 5.1 shows the weighted PMF curves for glycine on all three cement types.

_Figure 5.1: PMF curves for glycine on three cement types (Route 1 weighted average). OPC shows the deepest adsorption minimum, while Fly Ash shows the shallowest, consistent with the higher C3A content (35%) of Fly Ash cement reducing overall binding affinity._

The glycine PMF comparison shows that OPC binds glycine most strongly, followed by Slag, then Fly Ash. This ordering directly reflects the C3A weight: C3A is the weakest-binding phase for glycine, so higher C3A content in Fly Ash (35% vs 10% in OPC) reduces the composite binding energy. Slag cement, with intermediate C3A content (20%), falls between the two.

_Figure 5.2: Adsorption energies for five key biomolecules across three cement types. OPC shows the strongest tryptophan binding; Fly Ash shows enhanced DMEP binding relative to OPC and Slag, reflecting its higher C3A content and C3A's selective affinity for phosphate groups._

|     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
| **Rank** | **OPC Binder** | **Energy** | **Fly Ash Binder** | **Energy** | **Slag Binder** | **Energy** |
| 1   | TRPSCA-JS | \-19.65 | DMEP-AC | \-17.63 | DMEP-AC | \-16.45 |
| 2   | DMEP-AC | \-15.99 | PHO-AC | \-16.20 | TRPSCA-JS | \-16.38 |
| 3   | DMEP-JS | \-15.72 | DMEP-JS | \-15.71 | DMEP-JS | \-15.22 |
| 4   | PHO-JS | \-15.72 | PHO-JS | \-15.71 | PHO-JS | \-15.22 |
| 5   | ARGSCA-JS | \-14.67 | TRPSCA-JS | \-15.57 | PHO-AC | \-14.77 |

_Table 5.2: Top 5 binders for each cement type (Route 1, adsorption energies in kJ/mol). The shift from aromatic (TRP) to phosphate (DMEP, PHO) dominance in Fly Ash reflects its higher C3A content._

### Effect of C3A Content: Extended Compositional Sweep

[Section 6.5](#_6.5_Future_Directions) identifies a continuous composition sweep as a natural extension of the three-point OPC/Fly Ash/Slag comparison in [Section 5.2](#_5.2_Cement_Type). A limited version of this sweep was implemented here as a linear extension of the Route 1 approach: rather than re-running the full weighted-PMF-curve calculation ([Section 3.7](#_3.7_Route_1:)) at each composition, the per-phase adsorption energy minima (single model, Table 4.4 and Appendix A) are combined directly using the same volume-fraction weights, with the non-C3A phases held at their OPC ratio (C3S:C2S:C4AF = 0.65:0.15:0.10) while C3A is varied from 5% to 40%. This minima-weighted approximation is not identical to Route 1 proper, since averaging the minima of several curves is not the same as taking the minimum of the averaged curve, but it reproduces the three known Route 1 data points closely: for tryptophan, the approximation gives -19.9, -18.9 and -17.4 kJ/mol at the OPC, Slag and Fly Ash C3A fractions respectively, against the -19.65, -16.38 and -15.57 kJ/mol reported by the full Route 1 calculation in Table 5.2 ([Section 5.2](#_5.2_Cement_Type)) - the same ranking and a comparable magnitude, though Route 1 proper shows a larger swing between formulations. It should therefore be read as a qualitative screening tool for identifying promising compositions to re-run through the full pipeline, not as a replacement for Route 1 itself.

_Figure 5.3: Linear minima-weighted estimate of adsorption energy as a function of C3A volume fraction, for glycine, tryptophan, arginine and DMEP (phosphate analogue). Vertical lines mark the C3A content of the three cement formulations studied via full Route 1 in Section 5.2._

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **Composition (C3A %)** | **GLY** | **TRP** | **ARG** | **DMEP** |
| OPC (10%) | \-8.40 | \-19.86 | \-14.82 | \-16.32 |
| Slag (20%) | \-8.24 | \-18.87 | \-14.10 | \-16.03 |
| Fly Ash (35%) | \-8.01 | \-17.37 | \-13.03 | \-15.59 |

_Table 5.3: Linear minima-weighted adsorption energy estimates (kJ/mol) at the three studied cement compositions, for comparison against the full Route 1 values in Table 5.2._

All four molecules show the same monotonic trend- adsorption weakens as C3A content increases, consistent with C3A being the weakest-binding individual phase for every molecule considered except the phosphate analogues, where its relative weakening is smaller. This supports extending the pipeline to a genuine continuous sweep (full Route 1 weighted-PMF-curve calculation at, e.g., 5% C3A increments) as the highest-value next step for the compositional analysis.

Note: values in this table are from single-model predictions (ensemble model 1) rather than ensemble averages, for consistency with the per-phase single-model results presented in sections 4.2 to 4.5.

The most notable finding from the cement type comparison is the inversion in binding preference between OPC and Fly Ash cement. In OPC, tryptophan is the strongest binder (-19.65 kJ/mol) while DMEP ranks second (-15.99 kJ/mol). In Fly Ash cement, DMEP becomes the top binder (-17.63 kJ/mol) while tryptophan drops to fifth (-15.57 kJ/mol). This inversion is linked to the Fly Ash model's higher C3A weighting (35% versus 10% in OPC). C3A shows distinct selectivity for phosphate-containing molecules over aromatic amino acids: on C3A, DMEP (-16.4 kJ/mol) binds far more strongly than TRP (-10.9 kJ/mol). On C3S, the relationship reverses (TRP: -23.4, DMEP: approximately -12 kJ/mol).

This dependency on composition shift in binding selectivity has potential toxicological implications. Phospholipids - the primary element of cellular membranes and pulmonary surfactant - contain phosphate headgroups chemically like DMEP and PHO. If the Fly Ash model's altered phase composition leads to preferential adsorption of phosphate containing molecules, this could translate to enhanced disruption of pulmonary surfactant layers and greater membrane damage compared to OPC, providing a mechanistic basis for differential toxicity between cement formulations (Liu et al., 2020; Nel et al., 2009; Rouse et al., 2021).

## 5.3 Route 2: Patched Composite Surface

Following the revision described in [Section 3.8](#_3.8_Route_2:), surface potential generation (_GenerateSurfacePotentials.py_) completed successfully for all six probe types on the 576-atom PatchedOPCcement composite. However, inspection of the resulting probe potential files revealed that the fundamental non-convergence problem persisted even in this thin-layer geometry: the water probe potential reaches -103.5 kJ/mol at 2.06 nm separation, essentially unchanged from the -100 kJ/mol values observed for the original 2016-atom attempt, and still far from the near-zero convergence shown by single-phase surfaces at the same distance. This confirms that the non-convergence is not solely a consequence of the large z-extent addressed in the first revision but reflects a more basic limitation of the thin composite surface geometry itself.

As a direct consequence, the subsequent hypergeometric expansion step (_HGExpandSurfacePotential.py_) failed with an array indexing error (_IndexError: index -1 is out of bounds for axis 0 with size 0_) when applied to the 576-atom surface, identical in nature to the failure on the original 2016-atom version. The HGE expansion code's getValidRegion function expects probe potentials that converge towards near-zero at large distances (as verified in [Section 3.8](#_3.8_Route_2:) for the initial attempt) and cannot process the persistently non-convergent profiles produced by either version of the composite surface.

The root cause of this non-convergence is that the thin 0.5 nm surface patches lack the bulk atomic mass below the surface layer that is present in the full slab models. In the full slabs, the bulk atoms screen long-range electrostatic interactions, causing the probe potential to decay to near-zero within 1.5 nm. The thin patches do not provide this screening, leaving uncompensated long range Coulomb tails that persist at large probe-surface distances. Several solutions were identified for future work: using full slab depths with modified z-coordinate alignment to normalise the reference frame, applying an explicit long-range correction to enforce convergence of the probe potentials or bypassing atomic coordinate combination entirely and instead combining the per-phase HGE coefficient vectors as a weighted average before input to the neural network. The last of these approaches would be computationally trivial and would directly address whether the neural network produces different outputs for a composite descriptor versus a weighted average of phase descriptors.

## 5.4 Comparison of Route 1 and Route 2

A direct quantitative comparison of Route 1 and Route 2 adsorption energies was not achievable within the timeframe of this project due to the technical limitation described in [Section 5.3](#_5.3_Route_2:). Nevertheless, the physical principles underlying the two approaches can be discussed to frame the significance of the comparison.

Route 1 assumes that a biomolecule interacts with each mineral phase as if it were a pure homogeneous surface, and that the interactions are additive and linearly superimposable according to the phase volume fractions. This treats the mineral volume fraction as a proxy for the relative contribution of each phase to the overall interaction profile. Volume fraction and exposed surface area fraction are not necessarily equivalent, though: particle morphology, grain boundary distribution and differential exposure of mineral phases at the particle surface may cause the true surface area fractions to differ from the bulk volume fractions used here. The weighted average should therefore be understood as a composition-weighted approximation rather than a direct physical model of surface exposure. Route 2 allows the probe to encounter the combined electrostatic and van der Waals environment of laterally near mineral patches simultaneously, potentially capturing cooperative or competitive effects at mineral boundaries.

For the small molecules studied here (amino acid side chain analogues, 0.5-1 nm in size), the mineral patch dimensions in the composite surface (approximately 2-8 nm in x for the proportional patches) are large compared to the molecule. Most small molecules would interact predominantly with a single mineral phase at any given time, and the Routes would be expected to give similar results. The Route 1 weighted average thus may be a good approximation for small molecule adsorption on cement surfaces of this patch scale. For larger biomolecules such as intact proteins (5-20 nm hydrodynamic radius), the situation would be qualitatively different. A protein spanning a mineral boundary would experience the electrostatic environment of two phases simultaneously, and the effective surface chemistry presented to it would not simply be the linear combination of the individual phase chemistries. In this regime, Route 2 would be expected to diverge from Route 1, potentially capturing synergistic binding effects not accessible to the weighted average. This scale dependence of the Route 1 vs Route 2 difference is an important consideration for future work applying the pipeline to protein-scale biomolecule corona formation on cement surfaces.

# Chapter 6: Discussion and Conclusion

## 6.1 Physical Interpretation of Results

The predicted PMF landscapes across the four cement mineral phases reveal a coherent and physically interpretable picture of surface-biomolecule interactions in the cement system. Three distinct interaction mechanisms dominate across the four phases and biomolecule classes studied.

First, π-cation interactions between aromatic amino acids and Ca<sup>2+</sup> surface sites drive the strong and consistent binding of tryptophan, tyrosine, and phenylalanine across all four phases. This mechanism is particularly pronounced on C3S (the most calcium-dense phase) where tryptophan reaches -23.4 kJ/mol. The π-cation interaction between aromatic rings and divalent calcium is well-established in protein crystal structures, where Ca<sup>2+</sup> frequently coordinates with aromatic residues in calcium-binding proteins (Dougherty, 1996). The prediction that this interaction is important at calcium silicate mineral surfaces is chemically consistent.

Second, the C4AF surface shows the strongest overall binding affinity among the four phases, with particularly exceptional values for phosphate analogues and acidic amino acids (ensemble mean -29.50 kJ/mol for PHO-AC, single-model value -34.4 kJ/mol). The Brownmillerite structure nominally contains Fe<sup>3+</sup> Lewis acid centres, known to form strong coordination complexes with oxygen donor ligands including carboxylates and phosphates. However, it should be kept in mind that the C4AF model used here assigns Al<sup>3+</sup> to 76% of the crystallographically mixed sites, meaning the surface is aluminate in character. The strong binding observed most likely reflects a combination of coordination at the minority Fe<sup>3+</sup> sites and the broader electrostatic character of the Brownmillerite surface. This model does not permit decomposition of the total adsorption energy into separate Fe and Al contributions, so the relative importance of each cannot be established from the present calculations. The predicted adsorption energies for PHO and DMEP on C4AF are well below -20 kJ/mol. As noted in Section 4.5, values this strongly negative point to strong predicted thermodynamic binding, but confirming irreversibility still requires kinetic or experimental follow-up.

Third, the contrast between C3A and the other three phases reveals the importance of surface Lewis acid site identity. C3A's Al<sup>3+</sup> centres are weaker Lewis acids than Fe<sup>3+</sup> in C4AF and interact more weakly with most biological functional groups, producing the systematically lower adsorption energies observed across the amino acid library. However, C3A's selectivity for phosphate molecules (DMEP: -16.4 kJ/mol versus -10.9 for TRP) shows that Al<sup>3+</sup> does engage strongly with high-charge-density phosphate anions, a well-known feature of aluminium surface chemistry in environmental science contexts.

## 6.2 Implications for Cement Dust Toxicology

The computational results allow several testable predictions about the toxicological behaviour of cement dust. The strong and broadly consistent binding of aromatic amino acids (TRP, TYR, PHE) and charged residues (ARG, HIS, LYS) across all four cement phases suggests that the molecular functional groups represented by these residues may contribute to protein adsorption on cement surfaces, motivating future studies using intact proteins and competitive corona formation assays rather than isolated side chain analogues. Strong adsorption events can lead to protein unfolding, loss of biological function, and exposure of hydrophobic regions that trigger macrophage recognition and inflammatory cytokine release.

The cement type comparison result has potentially important public health implications. The predicted shift in binding selectivity from aromatic amino acids (OPC) to phosphate-containing molecules (Fly Ash) as C3A content increases suggests that Fly Ash cement dust may preferentially interact with pulmonary surfactant phospholipids. Pulmonary surfactant is rich in phosphatidylcholine, with dipalmitoyl phosphatidylcholine (DPPC) as a major molecular species. It lines the alveolar surface and reduces surface tension to prevent alveolar collapse. Adsorption of surfactant onto inhaled particles is a known mechanism of particle-induced lung injury; the prediction that Fly Ash may show enhanced phospholipid affinity compared to OPC provides a testable mechanistic hypothesis for differential toxicity between cement formulations.

The strong phosphate binding on C4AF (ensemble mean -29.50 kJ/mol for PHO-AC, single-model value -34.4 kJ/mol) suggests that even the minority C4AF fraction (10% in OPC) may make a disproportionately large contribution to phospholipid adsorption from pulmonary surfactant, given the thermodynamic advantage of >15 kJ/mol over other phases. This prediction - that the minor C4AF component may have disproportionate biological importance - is a specific, experimentally testable hypothesis that emerges from the computational analysis and would require explicit MD simulation or experimental surface tensiometry to validate.

### Comparison to Other Nanomaterial Classes (Plastics and Noble Metals)

Placing the cement results in the context of other nanomaterial classes studied by the same PMF-based methodology helps calibrate whether the binding strengths reported here are large or small in absolute terms. Maloney (2025) applied the same PMFPredictor machine-learning toolkit used in this thesis to unmodified polypropylene (PP) nanoplastic surfaces and reports predicted minima for tryptophan and tyrosine side-chain analogues of approximately -5 kBT (~-12.4 kJ/mol at 298 K). This is directly comparable in both magnitude and methodology (same neural network toolkit, same molecule library convention) to the weaker cement phases found here (TRP on C2S: -10.5 kJ/mol; TRP on C3A: -10.9 kJ/mol), but only about half the strength of the C3S and C4AF result (-23.4 and -19.9 kJ/mol respectively). This suggests that, at least for aromatic amino acids, the calcium- and iron-rich cement phases present a substantially more reactive surface than an unmodified plastic nanoparticle of comparable methodology.

For the phosphate-selective binding that distinguishes C4AF and C3A in this work, Simonetti et al. (2025) report DFT-computed binding energies for perfluorooctane sulfonate (PFOS), a molecule with a similar anionic head group to the DMEP/PHO analogues used here, of approximately -18 kcal/mol (~-75 kJ/mol) for a full PP nanoparticle and -5.5 kcal/mol (~-23 kJ/mol) for a single PP monomer. These are gas-phase DFT electronic energies rather than solution-phase PMFs and are not directly comparable in absolute terms, but the PP-monomer value (~-23 kJ/mol) is of the same order as the C3A/C2S phosphate result found here (~-13 to -16 kJ/mol) and well below the C4AF result (-29.5 to -34.4 kJ/mol), consistent with C4AF being an unusually strong phosphate binder even against a DFT-level plastic benchmark.

A direct comparison against noble-metal nanoparticles is harder to make quantitatively from the literature surveyed for this thesis. Subbotina, Rouse and Lobaskin (2023) applied a closely related PMF-based approach to PEGylated core-shell gold and silver nanoparticles and would be the natural source of comparable ΔG<sub>min</sub> values for gold and silver surfaces. Qualitatively, however, noble-metal corona formation in that lineage of work (see also Mosaddeghi Amini et al., 2023, 2024, for iron surfaces) is typically dominated by a small number of very strong individual binders (analogous to the C4AF phosphate result here) rather than broad, uniform affinity across the amino acid library, a pattern that appears qualitatively consistent with the cement results.

Overall, these results show that cement clinker phases can be moderate to strong binding surfaces. For aromatic amino acids, their binding strength is equal, or slightly stronger than, that of unmodified plastic nanoparticles. The particularly strong phosphate binding predicted for C4AF is also comparable to the DFT-level plastic benchmarks, despite the major difference in surface chemistry between the two materials. These results support the toxicological relevance discussed in [Section 6.2](#_6.2_Implications_for) and suggest that cement dust should not be considered biologically inert simply because it is a mineral rather than an engineered nanomaterial.

## 6.3 Computational Efficiency and Scalability

Direct MD-based PMF calculation via umbrella sampling typically requires hundreds to thousands of nanoseconds of cumulative sampling per surface-molecule pair, making exhaustive evaluation of 356 surface-molecule combinations computationally demanding on conventional hardware. By comparison, the PMFPredictor ensemble generated the corresponding 356 predictions, across all 10 bootstrap models, in approximately 2 to 3 minutes on a single CPU core, after the one-time cost of surface potential generation (13.5 hours total across the four phases).

The pipeline is highly scalable. Adding a new surface requires only the surface potential generation step (hours of compute) and the NPtoCSV conversion, the HGE expansion and prediction then run in minutes. Adding a molecule requires only the chemical HGE expansion (seconds per molecule). The entire framework can in principle be extended to any material for which CHARMM-GUI or a compatible crystallographic structure and force field are available.

## 6.4 Limitations and Sources of Uncertainty

Several limitations of the current approach should be acknowledged. The most substantial is the extrapolation of the PMFPredictor neural network beyond its training distribution. The model was trained on a set of inorganic surfaces - gold, silver, aluminium, copper, iron, TiO2, ZnO, graphene, graphene oxide, SiO2, and hydroxyapatite - that does not include cement mineral phases. The predictions for cement surfaces hence represent extrapolation, and their quantitative accuracy cannot be guaranteed without explicit MD-based validation. The qualitative trends and the relative ordering of adsorption energies across phases are more likely to be reliable than the absolute values.

Second, the crystal structure simplifications have known limitations. The C3S and C3A slab models represent single crystallographic surfaces ((001) and (010) respectively), while real cement particles expose multiple faces with different local compositions and charge densities. Multiple surface realisations with different terminations, hydroxylation states, and defect densities would be needed to fully characterise the surface interaction landscape.

Third, the C4AF disordered site treatment assigns each Al/Fe position to the majority element (Al), systematically underrepresenting the contribution of Fe<sup>3+</sup> sites which occupy 24% of the affected positions. A more accurate treatment would generate an ensemble of structures with statistically sampled Al/Fe distributions and average the resulting PMF predictions - an extension straightforward to implement within the current framework.

Fourth, the cement volume fractions used for OPC (C3S:65%, C2S:15%, C3A:10%, C4AF:10%), Fly Ash (C3S:30%, C2S:20%, C3A:35%, C4AF:15%), and Slag (C3S:40%, C2S:30%, C3A:20%, C4AF:10%) represent typical literature values from XRF and XRD analysis of commercial products, but actual compositions vary between manufacturers, geographic sources, and production conditions. The pipeline can be trivially re-run with experimentally determined compositions for specific materials of interest.

Fifth, the force field parameterisation of C2S and C4AF differs in an important way from that of C3S and C3A. The CHARMM-GUI-generated slabs for C3S and C3A use environment-specific INTERFACE FF atom types that distinguish, for example, calcium in silicate environments (ICA_S) from calcium in aluminate environments (ICA_A), and bridging oxygen (IOC1) from non-bridging oxygen (IOC2). The CIF-derived C2S and C4AF slabs use element-level parameter assignment, giving all calcium atoms the same parameters regardless of their local coordination environment. This is a qualitative simplification that may affect the surface potential calculations for these two phases. The impact is difficult to quantify without reference data, but it represents a systematic difference in the physical description of C2S and C4AF relative to C3S and C3A.

## 6.5 Future Directions

The most immediately impactful extension would be experimental validation of the PMF predictions for at least one cement mineral phase via either direct AFM force measurements or implicit-solvent MD simulation. The PMFPredictor repository includes pre-computed PMF data for TricalciumSilicate001 (identical to the C3S phase studied here) against a subset of amino acids, providing a direct comparison baseline for the predictions generated in this work. Extension to hydrated cement phases would highly improve the physical realism of the models. The primary hydration product of C3S and C2S is calcium silicate hydrate (C-S-H), an amorphous calcium silicate gel that constitutes most of the hardened cement paste. C-S-H surfaces are the relevant contact material in most exposure scenarios (hardened concrete dust) rather than the anhydrous clinker phases studied here. CIF structures for C-S-H analogue phases (tobermorite, jennite) are available in the COD and could be processed through the same pipeline. The multi-cement comparison could be extended to a continuous composition space by parameterising the Route 1 weighted average as a function of the four phase fractions, generating a composition-adsorption energy landscape across the full range of cement formulations. This would enable direct identification of cement compositions that minimise predicted binding of biologically harmful molecules - a potential rational design tool for low-toxicity cement formulations.

## 6.6 Conclusion

This thesis set out to address a specific gap: the inability of existing computational toxicology tools to meaningfully characterise complex multi-mineral industrial particulates such as cement dust. The work demonstrates that this gap can be closed, at least partially, using a machine-learning accelerated PMF prediction approach applied to atomistic surface slab models constructed from crystallographic reference structures.

The most important scientific finding is that cement mineral phases are not interchangeable in their biological interaction profiles. In the single-model results, the strongest C4AF phosphate predictions are more than twice as negative as the corresponding C3S predictions. C3A, at the other extreme, shows almost no affinity for small amino acids while keeping significant phosphate selectivity. These differences are large enough to matter: a cement formulation with greater C4AF content would be expected to show stronger predicted affinity for phosphate-containing molecular groups representative of phospholipid headgroups than one dominated by C3S, and Fly Ash cement with its elevated C3A content shows a measurably different biomolecular binding profile from OPC.

The Route 1 weighted average approach provides a practical composition weighted approximation for combining these per-phase profiles into a formulation-level prediction. The fact that three cement types (OPC, Fly Ash, Slag) can be differentiated computationally in a matter of minutes, after initial surface preparation, opens the possibility of using computational screening to prioritise cement compositions for experimental evaluation.

Route 2 did not produce final predictions within the scope of this project. The technical challenges encountered, related to probe potential convergence in the thin composite surface geometry, have been characterised and directions for resolution identified. Whether Route 1 and Route 2 give meaningfully different answers for small molecules remains an open question, though physical arguments suggest the difference would be modest at the amino acid scale and potentially more significant for protein-sized biomolecules.

The pipeline is reproducible, documented, and extensible. Adding a new material requires only a crystal structure and force field parameters. The 13.5 hours of CPU time invested in surface potential generation for four cement phases yields predictions for 356 surface-molecule pairs in minutes. This balance of setup cost versus predictive throughput is precisely what is needed for practical industrial screening applications.

# References

1.  Air Quality Life Index (AQLI), Energy Policy Institute, University of Chicago. Global impact of PM2.5 on life expectancy. Available at: https://aqli.epic.uchicago.edu (Accessed August 2026).
2.  Bhatt, A., Priyadarshini, S., Mohanakrishnan, A.A., Abri, A., Sattler, M. & Techapaphawit, S. (2019). Physical, chemical, and geotechnical properties of coal fly ash: A global review. Case Studies in Construction Materials, 11, e00263. DOI: 10.1016/j.cscm.2019.e00263.
3.  Choi, Y.K., Kern, N.R., Kim, S., Kanhaiya, K., Afshar, Y., Jeon, S.H., Jo, S., Brooks, B.R., Lee, J., Tadmor, E.B., Heinz, H. & Im, W. (2022). CHARMM-GUI Nanomaterial Modeler for Modelling and Simulation of Nanomaterial Systems. Journal of Chemical Theory and Computation, 18, 479-493. DOI: 10.1021/acs.jctc.1c00996.
4.  Colville, A.A. & Geller, S. (1971). The Crystal Structure of Brownmillerite, Ca2FeAlO5. Acta Crystallographica B, 27, 2311-2315. DOI: 10.1107/S0567740871005734. COD ID: 1200009.
5.  Dougherty, D.A. (1996). Cation-pi interactions in chemistry and biology: a new view of benzene, Phe, Tyr, and Trp. Science, 271(5246), 163-168.
6.  Grazulis, S., Chateigner, D., Downs, R.T., Yokochi, A.F.T., Quiros, M., Lutterotti, L., Manakova, E., Butkus, J., Moeck, P. & Le Bail, A. (2009). Crystallography Open Database - an open-access collection of crystal structures. Journal of Applied Crystallography, 42, 726-729. DOI: 10.1107/S0021889809016690.
7.  Heinz, H., Lin, T.J., Mishra, R.K. & Emami, F.S. (2013). Thermodynamically Consistent Force Fields for the Assembly of Inorganic, Organic, and Biological Nanostructures: The INTERFACE Force Field. Langmuir, 29, 1754-1765. DOI: 10.1021/la3038846.
8.  Jain, A., Ong, S.P., Hautier, G., Chen, W., Richards, W.D., Dacek, S., Cholia, S., Gunter, D., Skinner, D., Ceder, G. & Persson, K.A. (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. APL Materials, 1, 011002. DOI: 10.1063/1.4812323.
9.  Jo, S., Kim, T., Iyer, V.G. & Im, W. (2008). CHARMM-GUI: A Web-based Graphical User Interface for CHARMM. Journal of Computational Chemistry, 29, 1859-1865. DOI: 10.1002/jcc.20945.
10. Kopac, T. (2021). Protein corona, understanding the nanoparticle-protein interactions and future perspectives: A critical review. International Journal of Biological Macromolecules, 169, 290-301. DOI: 10.1016/j.ijbiomac.2020.12.108.
11. Liu, Y., Wang, Y., Zhang, Y., et al. (2020). Pulmonary surfactant and its role in the pathogenesis of lung diseases. Frontiers in Physiology, 11, 684. DOI: 10.3389/fphys.2020.00684.
12. Liu, Y. et al. (2021). Current Strategies in Assessment of Nanotoxicity: Alternatives to In Vivo Animal Testing. International Journal of Molecular Sciences, 22(8), 4216.
13. Lopez, H. & Lobaskin, V. (2015). Coarse-grained model of adsorption of blood plasma proteins onto nanoparticles. The Journal of Chemical Physics, 143, 243138. DOI: 10.1063/1.4935827.
14. Lynch, I. & Dawson, K.A. (2008). Protein-nanoparticle interactions. Nano Today, 3, 40-47. DOI: 10.1016/S1748-0132(08)70014-8.
15. Maloney, B. (2025). Computational Modelling of Polypropylene Nanoparticle Interactions with Biomolecules. MSc Thesis, School of Physics, University College Dublin. Supervisor: V. Lobaskin.
16. Mosaddeghi Amini, P., Subbotina, J. & Lobaskin, V. (2023). Milk protein adsorption on metallic iron surfaces. Nanomaterials, 13(12), 1859. DOI: 10.3390/nano13121859.
17. Mosaddeghi Amini, P., Rouse, I., Subbotina, J. & Lobaskin, V. (2024). Multiscale modelling of biomolecular corona formation on metallic surfaces. Beilstein Journal of Nanotechnology, 15, 215-229. DOI: 10.3762/bjnano.15.21.
18. Nel, A.E., Madler, L., Velegol, D., Xia, T., Hoek, E.M.V., Somasundaran, P., Klaessig, F., Castranova, V. & Thompson, M. (2009). Understanding biophysicochemical interactions at the nano-bio interface. Nature Materials, 8, 543-557. DOI: 10.1038/nmat2442.
19. Ong, S.P., Richards, W.D., Jain, A., Hautier, G., Kocher, M., Cholia, S., Gunter, D., Chevrier, V.L., Persson, K.A. & Ceder, G. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. Computational Materials Science, 68, 314-319. DOI: 10.1016/j.commatsci.2012.10.028.
20. Pietsch, C. et al. (2021). Characterization and Hazard Identification of Respirable Cement and Concrete Dust from Construction Activities. International Journal of Environmental Research and Public Health, 18(18), 9795.
21. Power, D., Rouse, I., Poggio, S., Brandt, E., Lopez, H., Lyubartsev, A. & Lobaskin, V. (2019). A multiscale model of protein adsorption on a nanoparticle surface. Modelling and Simulation in Materials Science and Engineering, 27, 084003. DOI: 10.1088/1361-651X/ab3b6e.
22. Puzyn, T., Leszczynska, D. & Leszczynski, J. (2009). Toward the development of "nano-QSARs": Advances and challenges. WIREs Nanomedicine and Nanobiotechnology, 1(2), 215-221.
23. Rouse, I., Power, D., Brandt, E.G., Schneemilch, M., Kotsis, K., Quirke, N., Lyubartsev, A.P. & Lobaskin, V. (2021). First principles characterisation of bio-nano interface. Physical Chemistry Chemical Physics, 23, 13473-13482. DOI: 10.1039/D1CP01116B.
24. Rouse, I. & Lobaskin, V. (2022). Computed surface and chemical potentials, expansion coefficients, structures, models and results for the PMFPredictor Toolkit. Zenodo. DOI: 10.5281/zenodo.7253566.
25. Rouse, I. & Lobaskin, V. (2023). Machine-learning based prediction of small molecule-surface interaction potentials. Faraday Discussions. DOI: 10.1039/D2FD00155A. Preprint: arXiv:2211.07999.
26. Shatkin, J. A., Thomas, R., Tolaymat, T., Wang, A., & Wong, J. (2013). A multi-stakeholder perspective on the use of alternative test strategies for nanomaterial safety assessment. Regulatory Toxicology and Pharmacology, 67(3), 337-345.
27. Shatkin, J. A., & Ong, K. J. (2016). Alternative Testing Strategies for Nanomaterials: State of the Science and Considerations for Risk Analysis. Risk Analysis, 36(8), 1564-1580.
28. Simonetti, F., Mancini, M., Gioia, V., Zumpano, R., Mazzei, F., Frugis, A. & Migliorati, V. (2025). Unveiling the adsorption mechanism of perfluorooctane sulfonate onto polypropylene nanoplastics: A combined theoretical and experimental investigation. Water Research, 278, 123324. DOI: 10.1016/j.watres.2025.123324.
29. Singh, M. and Garg, M. (2006). Developments in Portland slag cement. In: Advanced Concrete Technology: Constituent Materials (ed. J. Newman and B.S. Choo), Chapter 4, pp. 4-1 to 4-19. Butterworth-Heinemann, Oxford.
30. Stutzman, P. et al. (2015). Phase Analysis of Portland Cement by Combined Quantitative X-Ray Powder Diffraction and Scanning Electron Microscopy. NIST.
31. Subbotina, J., Rouse, I. & Lobaskin, V. (2023). In silico prediction of protein binding affinities onto core-shell pegylated noble metal nanoparticles for rational design of drug nanocarriers. Nanoscale, 15, 13371-13383. DOI: 10.1039/D3NR02026F.
32. Taylor, H.F.W. (1997). Cement Chemistry, 2nd edition. Thomas Telford, London. ISBN: 978-0727725929.
33. Toropov, A.A. & Toropova, A.P. (2015). QSPR/QSAR models for nanomaterials. Materials Science and Engineering: C, 51, 127-137.
34. Tsurumi, T., Hirano, Y., Kato, H., Kamiya, T. & Daimon, M. (1994). Crystal structure and hydration of belite. Ceramic Transactions, 40, 19-25. COD ID: 9012789.
35. World Health Organization (2021). WHO Global Air Quality Guidelines: Particulate Matter (PM2.5 and PM10), Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide. Geneva: WHO. ISBN: 978-92-4-003422-8.

# Appendix A: Complete Adsorption Energy Table

Table A.1 presents the predicted adsorption energies (kJ/mol) for all 89 molecules in the library across the four cement phases. The values shown are the global minima of the PMF curves generated using PMFPredictor ensemble model 1 and are included as a complete single-model reference dataset. Ensemble-averaged values used for quantitative comparisons are reported separately in Chapter 4 and Chapter 5. Molecules are listed by molecule code.

Note: Molecule naming follows the CHARMM36 residue convention. The suffix SCA indicates a side chain analogue (the amino acid side chain without the backbone). JS indicates Joung-Sponer partial charge parameterisation; AC indicates ACPYPE/AM1-BCC parameterisation. Where both JS and AC variants appear, they represent the same molecular structure with different charge models. DMEP is dimethyl phosphate (phospholipid headgroup analogue). PHO is a phosphate analogue. CHL is cholesterol. ETA is ethanolamine. EST is an ester analogue. DGL is galactose. BGLC is glucose. BGLCNA is N-acetylglucosamine. AMAN is mannose. MAMM is methylamine. MAS is methyl acetate. NC4 is tetramethylammonium. AFUC is fucose. ASPPSCA is aspartate phosphorylated side chain analogue. GLUPSCA is glutamate phosphorylated side chain analogue.

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **Molecule** | **C3S (kJ/mol)** | **C3A (kJ/mol)** | **C2S (kJ/mol)** | **C4AF (kJ/mol)** |
| AFUC-AC | \-12.0 | \-5.8 | \-4.9 | \-10.6 |
| AFUC-JS | \-11.1 | \-5.8 | \-5.0 | \-10.4 |
| ALASCA-AC | \-2.7 | \-0.2 | \-0.1 | \-0.6 |
| ALASCA-JS | \-0.6 | \-0.1 | \-0.1 | \-0.6 |
| AMAN-JS | \-9.7 | \-5.4 | \-5.7 | \-7.9 |
| ARGSCA-AC | \-15.6 | \-7.7 | \-6.7 | \-11.4 |
| ARGSCA-JS | \-18.3 | \-8.4 | \-7.2 | \-10.0 |
| ASNSCA-AC | \-5.6 | \-1.5 | \-2.4 | \-4.7 |
| ASNSCA-JS | \-5.9 | \-1.8 | \-2.7 | \-4.7 |
| ASPPSCA-AC | \-13.8 | \-7.1 | \-7.0 | \-13.9 |
| ASPPSCA-JS | \-12.9 | \-6.5 | \-6.6 | \-13.9 |
| ASPSCA-AC | \-4.6 | \-5.6 | \-4.7 | \-14.8 |
| ASPSCA-JS | \-5.4 | \-6.5 | \-5.2 | \-15.1 |
| BGLC-JS | \-13.5 | \-13.4 | \-7.8 | \-11.4 |
| BGLCNA-AC | \-13.6 | \-9.6 | \-7.4 | \-14.3 |
| BGLCNA-JS | \-16.9 | \-10.4 | \-10.6 | \-15.4 |
| BUTENE13-AC | \-3.4 | \-1.3 | \-1.3 | \-2.1 |
| BUTENE1-AC | \-3.0 | \-1.1 | \-1.1 | \-2.0 |
| BUTENE2-AC | \-3.1 | \-1.2 | \-1.1 | \-2.0 |
| CHL-AC | \-7.6 | \-3.7 | \-3.4 | \-7.2 |
| CHL-JS | \-7.2 | \-3.7 | \-3.5 | \-8.4 |
| CYMSCA-AC | \-14.5 | \-9.9 | \-7.5 | \-18.6 |
| CYMSCA-JS | \-16.0 | \-12.5 | \-9.4 | \-18.4 |
| CYSSCA-AC | \-3.7 | \-0.1 | \-0.1 | \-2.9 |
| CYSSCA-JS | \-4.4 | \-0.1 | \-0.1 | \-3.6 |
| DGL-AC | \-16.0 | \-10.5 | \-9.5 | \-11.7 |
| DGL-JS | \-13.5 | \-13.4 | \-7.8 | \-11.4 |
| DMEP-AC | \-16.5 | \-16.4 | \-15.2 | \-34.0 |
| DMEP-JS | \-16.3 | \-13.7 | \-13.1 | \-26.9 |
| EST-AC | \-8.2 | \-5.1 | \-4.2 | \-7.6 |
| EST-JS | \-7.8 | \-4.6 | \-3.8 | \-7.1 |
| ETA-AC | \-6.4 | \-3.5 | \-3.1 | \-7.4 |
| ETA-JS | \-6.6 | \-3.5 | \-3.3 | \-8.4 |
| ETHANE-AC | \-2.96 | \-0.11 | \-0.18 | \-1.40 |
| ETHENE-AC | \-2.05 | \-0.14 | \-0.13 | \-1.86 |
| GANSCA-AC | \-11.1 | \-5.9 | \-5.1 | \-8.4 |
| GANSCA-JS | \-10.6 | \-5.9 | \-4.8 | \-8.2 |
| GLNSCA-AC | \-9.4 | \-5.5 | \-4.8 | \-9.5 |
| GLNSCA-JS | \-10.4 | \-6.5 | \-5.8 | \-10.7 |
| GLUPSCA-AC | \-11.6 | \-6.1 | \-5.8 | \-12.6 |
| GLUPSCA-JS | \-10.8 | \-6.6 | \-6.1 | \-11.3 |
| GLUSCA-AC | \-9.6 | \-14.4 | \-13.1 | \-24.9 |
| GLUSCA-JS | \-9.2 | \-14.4 | \-9.3 | \-21.6 |
| GLY | \-9.0 | \-1.9 | \-5.0 | \-11.0 |
| GLY-AC | \-8.8 | \-6.5 | \-4.5 | \-10.3 |
| GLY-JS | \-8.5 | \-6.3 | \-4.7 | \-10.1 |
| HIDSCA-AC | \-11.9 | \-4.8 | \-4.3 | \-10.8 |
| HIDSCA-JS | \-11.1 | \-5.5 | \-6.0 | \-11.2 |
| HIESCA-AC | \-10.2 | \-3.4 | \-2.8 | \-12.3 |
| HIESCA-JS | \-10.8 | \-3.8 | \-3.4 | \-12.7 |
| HIPSCA-AC | \-18.3 | \-8.3 | \-5.8 | \-18.7 |
| HIPSCA-JS | \-16.8 | \-7.8 | \-5.7 | \-16.7 |
| HSPSCA-AC | \-18.3 | \-8.3 | \-5.8 | \-18.7 |
| HSPSCA-JS | \-16.8 | \-7.8 | \-5.7 | \-16.7 |
| ILESCA-AC | \-5.6 | \-3.4 | \-2.8 | \-5.9 |
| ILESCA-JS | \-5.1 | \-4.7 | \-3.3 | \-5.5 |
| LEUSCA-AC | \-5.9 | \-1.9 | \-4.4 | \-4.5 |
| LEUSCA-JS | \-5.9 | \-2.0 | \-4.4 | \-4.5 |
| LYSSCA-AC | \-11.3 | \-3.4 | \-5.1 | \-6.0 |
| LYSSCA-JS | \-11.3 | \-3.4 | \-6.6 | \-6.0 |
| MAMM-AC | \-6.3 | \-3.4 | \-2.7 | \-6.4 |
| MAMM-JS | \-6.2 | \-3.4 | \-2.7 | \-6.4 |
| MAS-AC | \-9.2 | \-4.0 | \-4.2 | \-8.0 |
| MAS-JS | \-8.5 | \-4.0 | \-4.1 | \-7.6 |
| METSCA-AC | \-9.2 | \-5.2 | \-4.4 | \-12.3 |
| METSCA-JS | \-9.2 | \-5.2 | \-4.4 | \-12.3 |
| NC4-AC | \-5.5 | \-1.6 | \-1.5 | \-4.4 |
| NC4-JS | \-5.3 | \-1.4 | \-1.3 | \-4.1 |
| PHESCA-AC | \-8.4 | \-1.6 | \-3.4 | \-11.7 |
| PHESCA-JS | \-8.4 | \-1.6 | \-3.4 | \-11.7 |
| PHO-AC | \-14.3 | \-16.0 | \-17.6 | \-34.4 |
| PHO-JS | \-16.5 | \-13.7 | \-13.1 | \-26.9 |
| PRO | \-4.0 | \-1.3 | \-0.9 | \-2.2 |
| PRO-AC | \-3.7 | \-0.8 | \-0.7 | \-2.0 |
| PRO-JS | \-3.7 | \-0.8 | \-0.7 | \-2.0 |
| PROPANE-AC | \-2.6 | \-0.5 | \-0.4 | \-1.4 |
| PROPENE-AC | \-3.53 | \-0.36 | \-0.57 | \-2.96 |
| PROSCA-AC | \-4.0 | \-1.3 | \-0.9 | \-2.2 |
| PROSCA-JS | \-4.0 | \-1.3 | \-0.9 | \-2.2 |
| SERSCA-AC | \-2.8 | \-0.2 | \-0.7 | \-1.6 |
| SERSCA-JS | \-2.8 | \-0.2 | \-0.7 | \-1.6 |
| THRSCA-AC | \-5.6 | \-2.1 | \-2.2 | \-3.4 |
| THRSCA-JS | \-5.6 | \-2.1 | \-2.2 | \-3.4 |
| TRPSCA-AC | \-16.7 | \-7.5 | \-5.8 | \-17.9 |
| TRPSCA-JS | \-23.4 | \-10.9 | \-10.5 | \-19.9 |
| TYRSCA-AC | \-17.6 | \-8.2 | \-4.1 | \-18.8 |
| TYRSCA-JS | \-15.9 | \-8.2 | \-4.1 | \-15.1 |
| VALSCA-AC | \-3.0 | \-0.7 | \-0.2 | \-2.2 |
| VALSCA-JS | \-3.0 | \-0.7 | \-0.2 | \-2.2 |

_Table A.1: Predicted adsorption energies (kJ/mol) for 89 molecules on four cement phases (ensemble model 1). Molecules listed by molecule code._

# Appendix B: Custom Python Scripts

This appendix presents the custom Python scripts developed for this project. All scripts were written for Python 3.10 and tested on Ubuntu 22.04 LTS.

## B.1 cif_to_pmf_csv.py

Converts CIF crystal structure files to the PMFPredictor surface CSV format. Used to generate C2S and C4AF surface slabs from Crystallography Open Database entries.

from pymatgen.core import Structure

import numpy as np

import pandas as pd

\# INTERFACE Force Field parameters by element

\# \[charge(e), mass(AMU), sigma(nm), epsilon(kJ/mol)\]

FF_PARAMS = {

'Ca': \[1.5, 40.0800, 2.98451070577e-01, 1.171520e+00\],

'Si': \[1.0, 28.0860, 4.00904423163e-01, 1.966480e+00\],

'Al': \[1.2, 26.9815, 4.09813410345e-01, 2.092000e+00\],

'Fe': \[1.2, 55.8450, 3.80000000000e-01, 1.000000e+00\],

'O': \[-1.0, 15.9994, 3.11814551349e-01, 3.347200e-01\],

}

def get_element(site):

"""Handle disordered sites by picking majority element."""

if site.is_ordered:

return site.specie.symbol

return max(site.species, key=lambda s: site.species\[s\]).symbol

def cif_to_csv(cif_file, output_name, outdir):

"""Convert CIF to PMFPredictor CSV format."""

structure = Structure.from_file(cif_file)

\# Build 3x3x2 supercell for adequate slab size

structure.make_supercell(\[3, 3, 2\])

output_path = f"{outdir}/{output_name}\_combined.csv"

count = 0

header = 'AtomID,AtomName,AtomType,x\[nm\],y\[nm\],z\[nm\],'

header += 'charge\[e\],mass\[AMU\],sigma\[nm\],epsilon\[kjmol\]'

with open(output_path, "w") as f:

f.write(header + "\\n")

for i, site in enumerate(structure):

elem = get_element(site)

if elem not in FF_PARAMS:

print(f"Warning: {elem} not in FF_PARAMS, skipping")

continue

charge, mass, sigma, epsilon = FF_PARAMS\[elem\]

\# Convert Angstrom to nm

x, y, z = site.coords / 10.0

count += 1

f.write(f"{count},{elem}{count},{elem},")

f.write(f"{x:.6f},{y:.6f},{z:.6f},")

f.write(f"{charge},{mass},{sigma},{epsilon}\\n")

print(f"Written {count} atoms to {output_path}")

\# Generate C2S and C4AF surface CSV files

cif_to_csv('C2S_beta.cif', 'DicalciumSilicate',

'PMFPredictor-Toolkit/Structures/Surfaces')

cif_to_csv('C4AF.cif', 'TetracalciumAluminoferrite',

'PMFPredictor-Toolkit/Structures/Surfaces')

## B.2a Single-model Route 1 script (weighted_cement_pmf.py)

Computes the Route 1 weighted average PMF for a given cement composition and extracts adsorption energies for all molecules in the library.

import numpy as np

import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

\# Path to PMFPredictor ensemble predictions

ENSEMBLE = (

'PMFPredictor-Toolkit/predicted_pmfs/'

'PMFPredictor-oct13-simplesplit-bootstrapped-ensemble1_canonical'

)

\# Cement formulation volume fractions

CEMENT_TYPES = {

'OPC': {

'TricalciumSilicate001': 0.65,

'DicalciumSilicate': 0.15,

'TricalciumAluminate010': 0.10,

'TetracalciumAluminoferrite': 0.10

},

'FlyAsh': {

'TricalciumSilicate001-new': 0.30,

'DicalciumSilicate': 0.20,

'TricalciumAluminate010': 0.35,

'TetracalciumAluminoferrite': 0.15

},

'Slag': {

'TricalciumSilicate001-new': 0.40,

'DicalciumSilicate': 0.30,

'TricalciumAluminate010': 0.20,

'TetracalciumAluminoferrite': 0.10

}

}

def compute_weighted_pmf(weights, mol):

"""Compute weighted average PMF for one molecule."""

weighted_pmf = None

r_range = None

for surface, weight in weights.items():

path = f"{ENSEMBLE}/{surface}/{mol}.dat"

data = np.loadtxt(path, delimiter=",", comments="#")

if weighted_pmf is None:

r_range = data\[:, 0\]

weighted_pmf = np.zeros_like(r_range)

weighted_pmf += weight \* data\[:, 1\]

return r_range, weighted_pmf

\# Get molecule list from first surface folder

first_surface = list(list(CEMENT_TYPES.values())\[0\].keys())\[0\]

molecules = \[

f.replace(".dat", "") for f in

os.listdir(f"{ENSEMBLE}/{first_surface}")

if f.endswith(".dat")

\]

\# Compute and report for each cement type

for cement_name, weights in CEMENT_TYPES.items():

results = \[\]

for mol in sorted(molecules):

try:

r, pmf = compute_weighted_pmf(weights, mol)

eads = np.min(pmf)

results.append((mol, eads, r, pmf))

except FileNotFoundError:

pass

results.sort(key=lambda x: x\[1\])

print(f"\\n=== {cement_name} Top 10 Binders ===")

for mol, eads, \_, _ in results\[:10\]:

print(f" {mol}: {eads:.2f} kJ/mol")

## B.2b Ensemble-averaged Route 1 script (weighted_cement_pmf_ensemble.py)

Computes the OPC weighted average adsorption energy independently for each of the 10 bootstrap ensemble models, then reports the mean and standard deviation across these 10 values. This is the corrected script that produced the ensemble results in Tables 4.1 to 4.3 and Table 5.1.

import numpy as np

import os

ENSEMBLE_BASE = 'PMFPredictor-Toolkit/predicted_pmfs'

ensembles = sorted(\[d for d in os.listdir(ENSEMBLE_BASE)

if d.startswith(

'PMFPredictor-oct13-simplesplit-bootstrapped-ensemble')\])

assert len(ensembles) == 10, (

f"Expected 10 bootstrap ensemble models, found {len(ensembles)}"

)

SURFACES = {

'TricalciumSilicate001-new': 'C3S',

'TricalciumAluminate010': 'C3A',

'DicalciumSilicate': 'C2S',

'TetracalciumAluminoferrite': 'C4AF'

}

OPC_WEIGHTS = {

'TricalciumSilicate001-new': 0.65,

'DicalciumSilicate': 0.15,

'TricalciumAluminate010': 0.10,

'TetracalciumAluminoferrite': 0.10

}

first = f'{ENSEMBLE_BASE}/{ensembles\[0\]}'

molecules = \[f.replace('.dat', '') for f in

os.listdir(f'{first}/TricalciumSilicate001-new')

if f.endswith('.dat')\]

results = \[\]

for mol in sorted(molecules):

\# Compute the OPC-weighted E_ads separately for each of the

\# 10 ensemble models, then take the mean and std of those

\# 10 independent values.

eads_per_model = \[\]

for ens in ensembles:

weighted_pmf = None

for surface, weight in OPC_WEIGHTS.items():

path = f'{ENSEMBLE_BASE}/{ens}/{surface}/{mol}.dat'

data = np.loadtxt(path, delimiter=',', comments='#')

if weighted_pmf is None:

weighted_pmf = np.zeros_like(data\[:, 1\])

weighted_pmf += weight \* data\[:, 1\]

eads_per_model.append(np.min(weighted_pmf))

eads_arr = np.array(eads_per_model)

results.append((mol, np.mean(eads_arr), np.std(eads_arr)))

results.sort(key=lambda x: x\[1\])

print('Top 10 OPC binders (10-model ensemble mean and std):')

for mol, mean_e, std_e in results\[:10\]:

print(f' {mol}: {mean_e:.2f} +/- {std_e:.2f} kJ/mol')

## B.3 patched_surface.py

Constructs the Route 2 spatially patched composite surface by tiling thin layers of each cement mineral phase in proportion to their volume fractions.

import pandas as pd

import numpy as np

\# OPC cement volume fractions

WEIGHTS = {

'TricalciumSilicate001-new': 0.65,

'DicalciumSilicate': 0.15,

'TricalciumAluminate010': 0.10,

'TetracalciumAluminoferrite': 0.10

}

SURFACE_DIR = 'PMFPredictor-Toolkit/Structures/Surfaces'

\# Load each phase and normalise z coordinates

\# So that z_max = 0 (surface at z=0, bulk below)

surfaces = {}

for name in WEIGHTS:

df = pd.read_csv(f"{SURFACE_DIR}/{name}\_combined.csv")

\# Shift z so surface top is at z=0

df\['z\[nm\]'\] = df\['z\[nm\]'\] - df\['z\[nm\]'\].max()

\# Take only the top 0.5 nm surface layer

df = df\[df\['z\[nm\]'\] >= -0.5\].copy()

surfaces\[name\] = df

print(f"{name}: {len(df)} atoms, "

f"z: {df\['z\[nm\]'\].min():.3f} to {df\['z\[nm\]'\].max():.3f}")

\# Target total atom count: 4x smallest phase

total_target = min(len(df) for df in surfaces.values()) \* 4

\# Build patches side by side in x direction

patches = \[\]

atom_id = 1

x_offset = 0.0

for name, weight in WEIGHTS.items():

df = surfaces\[name\].copy()

n_atoms = int(round(weight \* total_target))

\# Repeat atoms if phase has fewer than needed

if n_atoms > len(df):

repeats = (n_atoms // len(df)) + 1

df = pd.concat(\[df\] \* repeats, ignore_index=True)

df = df.head(n_atoms).copy()

\# Offset x coordinates to place patch beside previous

df\['x\[nm\]'\] = df\['x\[nm\]'\] - df\['x\[nm\]'\].min() + x_offset

x_offset = df\['x\[nm\]'\].max() + 0.5 # 0.5 nm gap

\# Renumber atom IDs

df\['AtomID'\] = range(atom_id, atom_id + len(df))

atom_id += len(df)

patches.append(df)

print(f"Added {len(df)} atoms from {name}")

\# Combine all patches

combined = pd.concat(patches, ignore_index=True)

print(f"Total atoms: {len(combined)}")

print(f"z range: {combined\['z\[nm\]'\].min():.3f} "

f"to {combined\['z\[nm\]'\].max():.3f} nm")

\# Save composite surface

out_path = f"{SURFACE_DIR}/PatchedOPCcement_combined.csv"

combined.to_csv(out_path, index=False)

print(f"Saved patched surface to {out_path}")
