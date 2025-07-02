
from typing import List, Union

import matplotlib.pyplot as plt

from src.performance.cumulative.cum_io import raw_to_smiles_batches
from src.performance.cumulative.investigator import CummulativeInvestigator
from src.performance.cumulative.projections import PCAProjector, SOAPProjector

FONT_TYPE = 'serif'
FONT_SIZE = 14
TITLE_FONT_SIZE = FONT_SIZE + 3

def make_scatter_plot(
    inv: CummulativeInvestigator,
    thresholds: List[int],
    tag: str='in_sample',
    projector: Union[PCAProjector, SOAPProjector]=None
) -> None:
    """ 
    Makes a scatter plot of 2D-projections of the discovered molecules.
    From the thresholds list, we can create batches of molecules to plot.
    Using different colors, we can plot all the newly discovered molecules from batch to batch.
    # TODO: For more meaningful latent space, use a pretrained property predictor to embed the molecules.
    """
    #projector = PCAProjector(mol_dataset=mol_dataset, model_dir=inv.model_dir)
    
    ref_smiles = projector._get_ref_smiles()

    
    # Split rollouts into batches
    thresholds = sorted(thresholds)

    # Make batched data
    smiles_sets, max_step_count = raw_to_smiles_batches(thresholds, inv.db_raw, tag=tag)

    # Retain only disjoint data
    disjoint_sets = {step: set() for step in thresholds}
    all_previous_sets = set()
    for step in thresholds:
        current_set = smiles_sets[step].copy()
        disjoint_set = current_set - all_previous_sets
        disjoint_sets[step] = disjoint_set
        all_previous_sets.update(current_set)


    # To 2d-space (only rediscovered molecules of course)
    projs_all = {step: projector(smiles_set=smiles_sets[step]) for step in smiles_sets} # only used for bar plot
    projs_disjoint = {step: projector(smiles_set=disjoint_sets[step]) for step in disjoint_sets}
    com_proj = projector(smiles_set=all_previous_sets, find_complement_set=True)

    print(f"com_proj.shape: {com_proj.shape}")
    
    scatter_size_rl = 2
    scatter_size_ref = 1

    # Labels for the plots
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    label_iter = iter(labels)
    def write_label(ax, label, height=1.10):
        ax.text(-0.10, height, label, transform=ax.transAxes, fontsize=FONT_SIZE+5, fontfamily=FONT_TYPE,
                va='center', ha='center', fontweight='bold')

    # Massive combined plot 
    # fig = plt.figure(figsize=(8, 8))
    # gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], wspace=0.4, hspace=0.5)

    n_plots = 3
    fig, ax = plt.subplots(1, n_plots, figsize=(13, 4.3), width_ratios=[1, 1, 1])
    # plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95)


    # Scatter plot - left column
    # ax1 = fig.add_subplot(gs[0:2, 0:2])

    ax1 = ax[0]

    ax1.scatter(com_proj[:, 0], com_proj[:, 1], color='gray', label='Complement Set', alpha=0.5, s=scatter_size_ref)

    # n_skips=1
    # interval = np.linspace(0.0, 0.75, len(thresholds)+n_skips)
    # for n_skip in range(n_skips):
    #     interval = np.delete(interval, 1)
    # colors = plt.cm.hsv(interval)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']


    for idx, (step, data) in enumerate(projs_disjoint.items()):
        if len(data) > 0:
            ax1.scatter(data[:, 0], data[:, 1], color=colors[idx], 
                        label=f'Threshold {step}', alpha=1., s=scatter_size_rl, edgecolors='w', linewidths=0.3)
        else:
            print(f"Step {step} has no data.")

    ax1.set_title('Rediscovered \n (SOAP + t-SNE)', fontsize=TITLE_FONT_SIZE, fontfamily=FONT_TYPE)
    ax1.set_xlabel('Projection 1', fontsize=FONT_SIZE+2, fontfamily=FONT_TYPE)
    ax1.set_ylabel('Projection 2', fontsize=FONT_SIZE+2, fontfamily=FONT_TYPE)
    # ax1.get_xaxis().get_offset_text().set_fontsize(FONT_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    #ax1.legend(loc='upper right')
    ax1.grid(True)
    #write_label(ax1, next(label_iter), height=1.05)

    # Bar plots
    num_bars = len(thresholds)
    x_positions = range(num_bars)

    thres_max = max(thresholds)
    # completion_ratio of last batch
    width = thresholds[-1] - thresholds[-2]
    missing_ratio = (thres_max - max_step_count) / width
    completion_ratio = 1-missing_ratio
    muliplier = 1 + missing_ratio/completion_ratio

    discovered = [len(smiles_sets[step]) for step in thresholds]
    discovered_disjoint = [len(disjoint_sets[step]) for step in thresholds]

    rediscovered = [len(projs_all[step]) for step in thresholds]
    rediscovered_disjoint = [len(projs_disjoint[step]) for step in thresholds]

    """
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title(f'Total (novel + rediscovered)', fontsize=TITLE_FONT_SIZE, fontfamily=FONT_TYPE)
    ax2.bar(x_positions, discovered, color=colors)
    #ax2.scatter(x_positions[-1], muliplier*discovered[-1], color='black', s=30)
    ax2.set_xticks([])
    write_label(ax2, next(label_iter))

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_title(f'Total disjoint', fontsize=TITLE_FONT_SIZE, fontfamily=FONT_TYPE)
    ax3.bar(x_positions, discovered_disjoint, color=colors)
    #ax3.scatter(x_positions[-1], muliplier*discovered_disjoint[-1], color='black', s=30)
    ax2.set_xticks([])
    write_label(ax3, next(label_iter))
    """

    # ax4 = fig.add_subplot(gs[2, 0])
    ax4 = ax[1]
    ax4.set_title(f'Rediscovered', fontsize=TITLE_FONT_SIZE, fontfamily=FONT_TYPE)
    ax4.bar(x_positions, rediscovered, color=colors)
    #ax4.scatter(x_positions[-1], muliplier*rediscovered[-1], color='black', s=30)
    ax4.set_xticks(x_positions)
    ax4.set_xticklabels(thresholds, rotation=45)
    #write_label(ax4, next(label_iter))
    ax4.tick_params(axis='both', which='major', labelsize=FONT_SIZE)


    # ax5 = fig.add_subplot(gs[2, 1])
    ax5 = ax[2]
    ax5.set_title(f'Rediscovered disjoint', fontsize=TITLE_FONT_SIZE, fontfamily=FONT_TYPE)
    ax5.bar(x_positions, rediscovered_disjoint, color=colors)
    #ax5.scatter(x_positions[-1], muliplier*rediscovered_disjoint[-1], color='black', s=30)
    ax5.set_xticks(x_positions)
    ax5.set_xticklabels(thresholds, rotation=45)
    #write_label(ax5, next(label_iter))
    ax5.tick_params(axis='both', which='major', labelsize=FONT_SIZE)



    # Uniqueness count plot - calculate the "novelty" of each batch, i.e. the number of molecules that are different from the union of all other batches

    ref_reduced_sets = {step: set([smiles for smiles in smiles_set if smiles in ref_smiles]) \
                            for step, smiles_set in smiles_sets.items()}

    # Loop over batches and compute the overlap counts
    unique_counts = []
    for i, step in enumerate(thresholds):
        other_steps = thresholds[:i] + thresholds[i+1:]
        other_set = set()
        for other_step in other_steps:
            other_set.update(ref_reduced_sets[other_step])

        # Calculate the number of unique molecules
        unique_counts.append(len(ref_reduced_sets[step] - other_set))

    """
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_title(f'Unique rediscovered', fontsize=TITLE_FONT_SIZE, fontfamily=FONT_TYPE)
    ax6.bar(x_positions, unique_counts, color=colors)
    #ax6.scatter(x_positions[-1], muliplier*unique_counts[-1], color='black', s=30)
    ax6.set_xticks(x_positions)
    ax6.set_xticklabels(thresholds, rotation=45)
    write_label(ax6, next(label_iter))
    """

    print(f"unique_counts: {unique_counts}")
    print(f"sum unique_counts: {sum(unique_counts)}")

    print(f"rediscovered disjoint: {rediscovered_disjoint}")
    print(f"sum rediscovered disjoint: {sum(rediscovered_disjoint)}")

    plt.tight_layout()  # Adjust spacing between subplots
    # plt.show()

    # Save the plot
    plot_path = inv.plot_dir / 'scatter_plot.png'
    plt.savefig(plot_path, dpi=300)
    print(f'Scatter plot saved to {plot_path}')

