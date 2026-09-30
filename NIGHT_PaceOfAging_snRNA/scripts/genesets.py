"""Hand-curated blood gene sets (from the literature, written from memory; used only as feature subsets)."""
SMOKING = ["AHRR", "LRRN3", "GPR15", "SASH1", "PID1", "CLDND1", "P2RY6", "FUCA1", "SEMA6B", "AARD", "CYP1B1",
           "PTGFRN", "MMP25", "TMEM163", "RGL1", "CLEC10A", "DSC2", "F2RL3", "CYP1A1", "SERPING1"]
AGING_PETERS_DOWN = ["CD248", "LRRN3", "NELL2", "ABLIM1", "MYC", "FLNB", "LEF1", "CCR7", "FAM102A", "SLC16A10",
                     "PDE9A", "BACH2", "NOG", "SERPINE2", "FCGBP", "MXRA8", "TCF7", "SELL", "CR2", "IGLL3P",
                     "BCL11B", "NT5E", "SATB1", "IL7R", "CD27", "CD28"]
AGING_PETERS_UP = ["GZMH", "GZMB", "KLRD1", "CX3CR1", "NKG7", "PRF1", "FGFBP2", "ADRB2", "TGFBR3", "B3GAT1",
                   "KLRG1", "ZEB2", "TBX21", "GNLY", "CCL5", "FCGR3A", "SPON2", "GZMA", "PLEK", "S1PR5"]
IFN = ["ISG15", "IFI44L", "IFI44", "IFI27", "MX1", "RSAD2", "OAS1", "OAS2", "OAS3", "IFIT1", "IFIT3", "IFI6",
       "HERC5", "EPSTI1", "XAF1", "STAT1", "IRF7", "SIGLEC1"]
INFLAM = ["IL1B", "TNF", "CXCL8", "NFKBIA", "S100A8", "S100A9", "S100A12", "IL6", "CCL3", "CCL4", "IER3", "FOS",
          "JUN", "DUSP1", "PTGS2", "SOCS3", "TNFAIP3", "CD14", "FCN1", "LYZ"]
SENESCENCE = ["CDKN1A", "CDKN2A", "CDKN2B", "SERPINE1", "GDF15", "TP53", "MDM2", "LMNB1", "IGFBP3", "IGFBP7",
              "TNFRSF10C", "B2M", "SESN1", "SESN2", "GADD45A"]
METABOLIC = ["TXNIP", "ARRDC4", "SLC2A1", "SLC2A3", "PDK4", "CPT1A", "ABCA1", "ABCG1", "LDLR", "HMGCR", "SREBF2",
             "INSR", "IRS2", "FKBP5", "TSC22D3", "PER1", "KLF9", "ZBTB16"]
ALL_SETS = {"smoking": SMOKING, "aging_down": AGING_PETERS_DOWN, "aging_up": AGING_PETERS_UP, "ifn": IFN,
            "inflam": INFLAM, "senescence": SENESCENCE, "metabolic": METABOLIC}
UNION = sorted(set(sum(ALL_SETS.values(), [])))
