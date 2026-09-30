#!/usr/bin/env python
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import os

saved_directory = "/home/nlapique/Blood/vo2max_modelvo2max"
os.makedirs(saved_directory, exist_ok=True)
# Load the data
print("Loading AnnData with denoised expression...")
adata = sc.read_h5ad('/home/nlapique/Blood/adata_hvg3000_with_mrvi_denoised_vo2max2.h5ad')

print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")

# Check the grouping variables
print("\nChecking 'VOmax26_bin_male' column...")
if 'VOmax26_bin_male' in adata.obs.columns:
    print(f"  Groups in VOmax26_bin_male: {adata.obs['VOmax26_bin_male'].unique()}")
    print(f"  Counts (including NA): {adata.obs['VOmax26_bin_male'].value_counts(dropna=False)}")
else:
    print("  ⚠ Warning: 'VOmax26_bin_male' not found!")
    print(f"  Available columns: {list(adata.obs.columns)}")

print("\nChecking 'CellType' column...")
if 'CellType' in adata.obs.columns:
    print(f"  Cell types: {adata.obs['CellType'].unique()}")
    print(f"  Counts:\n{adata.obs['CellType'].value_counts()}")
else:
    print("  ⚠ Warning: 'CellType' not found!")
    print(f"  Available columns: {list(adata.obs.columns)}")

# Verify we have the required columns
if 'VOmax26_bin_male' not in adata.obs.columns or 'CellType' not in adata.obs.columns:
    print("\nERROR: Required columns not found. Please check column names.")
    print(f"Available columns: {list(adata.obs.columns)}")
else:
    # *** FIX 1: REMOVE NA VALUES BEFORE ANALYSIS ***
    print("\n" + "="*70)
    print("PREPROCESSING: REMOVING NA VALUES")
    print("="*70)
    
    n_cells_before = adata.n_obs
    n_na = adata.obs['VOmax26_bin_male'].isna().sum()
    
    print(f"\nCells before filtering: {n_cells_before}")
    print(f"Cells with NA in VOmax26_bin_male: {n_na} ({100*n_na/n_cells_before:.1f}%)")
    
    # Filter out NA values
    adata_clean = adata[~adata.obs['VOmax26_bin_male'].isna()].copy()
    
    print(f"Cells after filtering: {adata_clean.n_obs}")
    print(f"Cells removed: {n_cells_before - adata_clean.n_obs}")
    
    # Verify we only have 2 groups now
    print(f"\nRemaining groups: {adata_clean.obs['VOmax26_bin_male'].unique()}")
    print(f"Group counts:\n{adata_clean.obs['VOmax26_bin_male'].value_counts()}")
    
    # Prepare data for DE analysis
    print("\n" + "="*70)
    print("DIFFERENTIAL EXPRESSION ANALYSIS")
    print("="*70)
    
    adata_de = adata_clean.copy()
    # Use denoised expression
    adata_de.X = adata_de.layers['mrvi_denoised'].copy()
    # Log transform
    sc.pp.log1p(adata_de)
    
    #########################################################################
    # ANALYSIS 1: ALL CELL TYPES MERGED
    #########################################################################
    print("\n" + "="*70)
    print("ANALYSIS 1: ALL CELL TYPES MERGED")
    print("="*70)
    
    print("\nRunning DE analysis for VOmax26_bin_male (all cells)...")
    print("  Control group (reference): VOmax26_q20_60_male")
    print("  Test group: VOmax26_q80_99_male")
    
    # *** FIX 2: EXPLICITLY SET REFERENCE GROUP ***
    # Wilcoxon test
    sc.tl.rank_genes_groups(
        adata_de,
        groupby='VOmax26_bin_male',
        method='wilcoxon',
        reference='VOmax26_q20_60_male',  # <-- EXPLICIT REFERENCE
        key_added='de_vomax_all',
        corr_method='benjamini-hochberg'
    )
    
    # Extract and save results
    # When using reference, only get results for the test group
    result_all = sc.get.rank_genes_groups_df(adata_de, group='VOmax26_q80_99_male', key='de_vomax_all')
    result_all.to_csv(
    f'{saved_directory}/DE_VOmax_all_celltypes_merged.csv',
    index=False
)
    print("  ✓ Saved: /home/nlapique/Blood/DE_VOmax_all_celltypes_merged.csv")
    
    # Show top genes (positive logFC = higher in high VO2max)
    print(f"\nTop 10 DE genes (VOmax26_q80_99_male vs VOmax26_q20_60_male):")
    print("\nHighest in High VO2max group (positive logFC):")
    top_up = result_all.nlargest(10, 'logfoldchanges')
    for idx, row in top_up.iterrows():
        print(f"  {row['names']}: logFC={row['logfoldchanges']:.2f}, "
              f"adj_pval={row['pvals_adj']:.2e}")
    
    print("\nHighest in Normal VO2max group (negative logFC):")
    top_down = result_all.nsmallest(10, 'logfoldchanges')
    for idx, row in top_down.iterrows():
        print(f"  {row['names']}: logFC={row['logfoldchanges']:.2f}, "
              f"adj_pval={row['pvals_adj']:.2e}")
    
    #########################################################################
    # ANALYSIS 2: SEPARATE FOR EACH CELL TYPE
    #########################################################################
    print("\n" + "="*70)
    print("ANALYSIS 2: SEPARATE ANALYSIS FOR EACH CELL TYPE")
    print("="*70)
    
    cell_types = adata_de.obs['CellType'].unique()
    print(f"\nAnalyzing {len(cell_types)} cell types...")
    
    all_results = []
    
    for cell_type in cell_types:
        print(f"\n  Processing: {cell_type}...")
        
        # Subset to this cell type
        adata_ct = adata_de[adata_de.obs['CellType'] == cell_type].copy()
        
        # Check if we have both groups
        groups_in_ct = adata_ct.obs['VOmax26_bin_male'].unique()
        if len(groups_in_ct) < 2:
            print(f"    ⚠ Skipping {cell_type}: only {len(groups_in_ct)} group(s) present")
            continue
        
        # Check sample sizes
        group_counts = adata_ct.obs['VOmax26_bin_male'].value_counts()
        print(f"    Sample sizes: {dict(group_counts)}")
        
        if any(group_counts < 3):
            print(f"    ⚠ Warning: Small sample size in {cell_type}")
        
        try:
            # *** FIX 3: EXPLICIT REFERENCE FOR EACH CELL TYPE ***
            # Run DE analysis
            sc.tl.rank_genes_groups(
                adata_ct,
                groupby='VOmax26_bin_male',
                method='wilcoxon',
                reference='VOmax26_q20_60_male',  # <-- EXPLICIT REFERENCE
                key_added=f'de_vomax_{cell_type}',
                corr_method='benjamini-hochberg'
            )
            
            # Extract results (only for test group when using reference)
            result_ct = sc.get.rank_genes_groups_df(
                adata_ct, 
                group='VOmax26_q80_99_male', 
                key=f'de_vomax_{cell_type}'
            )
            result_ct['CellType'] = cell_type
            all_results.append(result_ct)
            
            # Save individual cell type results
            output_file = (
            f'{saved_directory}/DE_VOmax_'
            f'{cell_type.replace(" ", "_").replace("/", "_")}.csv'
            )
            result_ct.to_csv(output_file, index=False)
            print(f"    ✓ Saved: {output_file}")
            
            # Show top significant genes
            sig_genes = (result_ct['pvals_adj'] < 0.05).sum()
            top_up = result_ct.nlargest(3, 'logfoldchanges')
            top_down = result_ct.nsmallest(3, 'logfoldchanges')
            print(f"      Significant DE genes: {sig_genes} (p<0.05)")
            print(f"      Top upregulated: {', '.join(top_up['names'].tolist())}")
            print(f"      Top downregulated: {', '.join(top_down['names'].tolist())}")
                
        except Exception as e:
            print(f"    ✗ Error in {cell_type}: {e}")
    
    # Combine all cell type results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(
        f'{saved_directory}/DE_VOmax_by_celltype_combined.csv',
        index=False
)


        print(f"\n  ✓ Saved combined results: /home/nlapique/Blood/DE_VOmax_by_celltype_combined.csv")
    
    #########################################################################
    # VISUALIZATION
    #########################################################################
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Figure 1: Overall results (all cell types merged)
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Differential Expression: VOmax26_q80_99_male vs VOmax26_q20_60_male (All Cell Types)', fontsize=16)
    
    # Volcano plot
    logfcs = result_all['logfoldchanges'].values
    pvals = result_all['pvals_adj'].values
    pvals_plot = -np.log10(pvals + 1e-300)
    
    axes1[0, 0].scatter(logfcs, pvals_plot, alpha=0.5, s=10, c='gray')
    sig_mask = pvals < 0.05
    axes1[0, 0].scatter(logfcs[sig_mask], pvals_plot[sig_mask], alpha=0.7, s=10, c='red')
    axes1[0, 0].axhline(-np.log10(0.05), color='blue', linestyle='--', label='p=0.05')
    axes1[0, 0].axvline(0, color='black', linestyle='-', linewidth=0.5)
    axes1[0, 0].set_xlabel('Log2 Fold Change (High vs Normal VO2max)')
    axes1[0, 0].set_ylabel('-log10(adjusted p-value)')
    axes1[0, 0].set_title(f'Volcano Plot')
    axes1[0, 0].legend()
    
    # Rank genes plot
    sc.pl.rank_genes_groups(adata_de, key='de_vomax_all', n_genes=20, sharey=False, ax=axes1[0, 1], show=False)
    
    # Top genes heatmap
    try:
        sc.pl.rank_genes_groups_heatmap(
            adata_de, 
            key='de_vomax_all', 
            n_genes=20,
            groupby='VOmax26_bin_male',
            show=False,
            ax=axes1[1, 0],
            cmap='RdBu_r'
        )
    except:
        axes1[1, 0].text(0.5, 0.5, 'Heatmap generation failed', ha='center', va='center')
    
    # Summary bar plot - upregulated vs downregulated
    n_sig_up = ((result_all['pvals_adj'] < 0.05) & (result_all['logfoldchanges'] > 0)).sum()
    n_sig_down = ((result_all['pvals_adj'] < 0.05) & (result_all['logfoldchanges'] < 0)).sum()
    
    axes1[1, 1].bar(['Higher in High VO2max', 'Higher in Normal VO2max'], 
                    [n_sig_up, n_sig_down], 
                    color=['#ff7f0e', '#1f77b4'])
    axes1[1, 1].set_ylabel('Number of DE genes (p < 0.05)')
    axes1[1, 1].set_title('Significant DE genes by direction')
    axes1[1, 1].text(0, n_sig_up + 5, str(n_sig_up), ha='center', va='bottom')
    axes1[1, 1].text(1, n_sig_down + 5, str(n_sig_down), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/home/nlapique/Blood/DE_VOmax_all_celltypes.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: /home/nlapique/Blood/DE_VOmax_all_celltypes.png")
    plt.close()
    
    # Figure 2: Summary across cell types
    if all_results and len(all_results) > 0:
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
        fig2.suptitle('Differential Expression: VOmax26_bin_male by Cell Type', fontsize=16)
        
        # Count significant genes per cell type
        sig_by_celltype = []
        for result_df in all_results:
            ct = result_df['CellType'].iloc[0]
            n_sig = (result_df['pvals_adj'] < 0.05).sum()
            sig_by_celltype.append({'CellType': ct, 'n_sig': n_sig})
        
        sig_df = pd.DataFrame(sig_by_celltype).sort_values('n_sig', ascending=False)
        
        axes2[0, 0].barh(sig_df['CellType'], sig_df['n_sig'])
        axes2[0, 0].set_xlabel('Number of DE genes (p < 0.05)')
        axes2[0, 0].set_ylabel('Cell Type')
        axes2[0, 0].set_title('Significant DE genes by cell type')
        
        # Top genes across cell types (most frequently DE)
        gene_freq = {}
        for result_df in all_results:
            top_genes = result_df[result_df['pvals_adj'] < 0.05].head(20)
            for gene in top_genes['names']:
                gene_freq[gene] = gene_freq.get(gene, 0) + 1
        
        if gene_freq:
            gene_freq_df = pd.DataFrame(list(gene_freq.items()), columns=['Gene', 'Frequency'])
            gene_freq_df = gene_freq_df.sort_values('Frequency', ascending=False).head(20)
            
            axes2[0, 1].barh(gene_freq_df['Gene'], gene_freq_df['Frequency'])
            axes2[0, 1].set_xlabel('Number of cell types where DE')
            axes2[0, 1].set_ylabel('Gene')
            axes2[0, 1].set_title('Top genes DE across multiple cell types')
            axes2[0, 1].invert_yaxis()
        
        # Distribution of log fold changes
        all_logfc = pd.concat([df[['logfoldchanges', 'CellType']] for df in all_results])
        axes2[1, 0].boxplot([all_results[i]['logfoldchanges'].values for i in range(len(all_results))],
                            labels=[all_results[i]['CellType'].iloc[0] for i in range(len(all_results))])
        axes2[1, 0].set_xlabel('Cell Type')
        axes2[1, 0].set_ylabel('Log2 Fold Change')
        axes2[1, 0].set_title('Distribution of log fold changes')
        axes2[1, 0].tick_params(axis='x', rotation=45)
        
        # Summary stats table
        summary_stats = []
        for result_df in all_results:
            ct = result_df['CellType'].iloc[0]
            n_total = len(result_df)
            n_sig = (result_df['pvals_adj'] < 0.05).sum()
            max_logfc = result_df['logfoldchanges'].abs().max()
            summary_stats.append([ct, n_sig, f'{max_logfc:.2f}'])
        
        axes2[1, 1].axis('tight')
        axes2[1, 1].axis('off')
        table = axes2[1, 1].table(
            cellText=summary_stats,
            colLabels=['Cell Type', 'DE genes', 'Max |logFC|'],
            loc='center',
            cellLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        axes2[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(
        f'{saved_directory}/DE_VOmax_all_celltypes.png',
        dpi=150,
        bbox_inches='tight'
        )
        print("  ✓ Saved: /home/nlapique/Blood/DE_VOmax_by_celltype_summary.png")
        plt.close()
    
    #########################################################################
    # SUMMARY
    #########################################################################
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*70)
    
    print("\n1. ALL CELL TYPES MERGED:")
    print(f"   Total genes analyzed: {len(result_all)}")
    print(f"   Reference group: VOmax26_q20_60_male (control)")
    n_sig_total = (result_all['pvals_adj'] < 0.05).sum()
    n_sig_up = ((result_all['pvals_adj'] < 0.05) & (result_all['logfoldchanges'] > 0)).sum()
    n_sig_down = ((result_all['pvals_adj'] < 0.05) & (result_all['logfoldchanges'] < 0)).sum()
    print(f"   Total significant DE genes: {n_sig_total} (adj p < 0.05)")
    print(f"     - Higher in High VO2max: {n_sig_up}")
    print(f"     - Higher in Normal VO2max: {n_sig_down}")
    
    print("\n2. BY CELL TYPE:")
    print(f"   Analyzed {len(all_results)} cell types")
    print(f"   Reference group: VOmax26_q20_60_male (control)")
    for result_df in all_results:
        ct = result_df['CellType'].iloc[0]
        n_sig = (result_df['pvals_adj'] < 0.05).sum()
        print(f"   {ct}: {n_sig} significant DE genes")
    
    print("\n3. OUTPUT FILES:")
    print("   - DE_VOmax_all_celltypes_merged.csv (all cells)")
    print("   - DE_VOmax_by_celltype_combined.csv (all cell types combined)")
    print("   - DE_VOmax_<CellType>.csv (individual cell types)")
    print("   - DE_VOmax_all_celltypes.png (visualization - all cells)")
    print("   - DE_VOmax_by_celltype_summary.png (visualization - by cell type)")
    
    print("\n✓✓✓ Differential expression analysis complete!")
    print("\nNote: Positive logFC = higher in VOmax26_q80_99_male (high VO2max)")
    print("      Negative logFC = higher in VOmax26_q20_60_male (normal VO2max)")