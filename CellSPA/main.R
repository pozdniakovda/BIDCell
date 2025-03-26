# Ensure that CellSPA and SingleCellExperiement are installed from BiocManager first

library(CellSPA)
library(SingleCellExperiment)
library(ggplot2)
library(glue)

data_dir <- "extdata/BIDCell_csv_output" # contains the CSV file with BIDCell output from model.predict()
tiff_path <- "extdata/BIDCell_output_subset.tif" # equivalent to *_connected.tif
sce_path <- "extdata/sce_FFPE_full.rds" # optionally substitute with default `system.file("extdata/sce_FFPE_full.rds", package = "CellSPA")`

message(glue("Data directory: {data_dir}"))
message(glue("Image path: {tiff_path}"))

message("Reading BIDCell data...")
spe <- readBIDCell(data_dir,
                   tiff_path = tiff_path,
                   method_name = "BIDCell",
                   spatialCoordsNames = c("cell_centroid_x",
                                          "cell_centroid_y"))

message("Processing SPE...")
spe <- processingSPE(spe,                                                         
                     qc_range = list(total_transciprts = c(20, 2000),
                                     total_genes = c(20, Inf)))

message("Generating polygons...")
spe <- generatePolygon(spe)

message("Calculating baseline metrics...")
spe <- calBaselineAllMetrics(spe, verbose = TRUE)

message("Fetching SCE FFPE reference...")
sce_ref_full <- readRDS(sce_path)
sce_ref <- processingRef(sce_ref_full, 
                         celltype = sce_ref_full$graph_cluster_anno, 
                         subset_row = rownames(spe))

message("Calculating expression correlation...")
spe <- calExpressionCorrelation(spe,                                  
                                sce_ref,
                                ref_celltype = sce_ref$celltype,
                                method = c("pearson", "cosine"),
                                spe_exprs_values = "logcounts",
                                ref_exprs_values = "mean")

message("Calculating aggregated expression correlation using reference mean...")
spe <- calAggExpressionCorrelation(spe,                           
                                   celltype = "mean_celltype_correlation",
                                   sce_ref = sce_ref,
                                   ref_celltype = "celltype",
                                   method = c("pearson"),
                                   spe_exprs_values = "logcounts",
                                   ref_exprs_values = "mean")

message("Calculating aggregated expression correlation using reference prop_detected...")
spe <- calAggExpressionCorrelation(spe,                           
                                   celltype = "mean_celltype_correlation",
                                   sce_ref = sce_ref,
                                   ref_celltype = "celltype",
                                   method = c("pearson"),
                                   spe_exprs_values = "logcounts",
                                   ref_exprs_values = "prop_detected")

message("Generating positive and negative marker lists...")
positive_marker_list <- generateMarkerList(sce_ref, type = "positive")
negative_marker_list <- generateMarkerList(sce_ref, type = "negative", t = 1)

message("Calculating marker purity...")
spe <- calMarkerPurity(spe,
                       celltype = "mean_celltype_correlation",
                       marker_list = positive_marker_list,
                       marker_list_name = "positive")
spe <- calMarkerPurity(spe,
                       celltype = "mean_celltype_correlation",
                       marker_list = negative_marker_list,
                       marker_list_name = "negative")

message("Calculating marker percent...")
spe <- calMarkerPct(spe,
                    celltype = "mean_celltype_correlation",
                    marker_list = positive_marker_list,
                    marker_list_name = "positive")
spe <- calMarkerPct(spe,
                    celltype = "mean_celltype_correlation",
                    marker_list = negative_marker_list,
                    marker_list_name = "negative")

message("Calculatinig spatial metrics diversity...")
spe <- calSpatialMetricsDiversity(spe, 
                                  celltype = "mean_celltype_correlation")

message("Calculatiing negative marker vs. dist...")
nn_celltype_pair <- c("B Cells", "CD4 T|CD8 T")
neg_markers <- list("B Cells" = c("CD3C", "CD3E", "CD8A"),
                    "CD4 T|CD8 T" = c("MS4A1", "CD79A", "CD79B"))
spe <- calNegMarkerVsDist(spe,
                          "mean_celltype_correlation",
                          nn_celltype_pair,
                          neg_markers)

message("Saving colData dataframe...")
col_df <- as.data.frame(colData(spe))
write.csv(col_df, file = "extdata/colData_spe.csv", row.names = TRUE)

message("Saving rowData dataframe...")
row_df <- as.data.frame(rowData(spe))
write.csv(col_df, file = "extdata/rowData_spe.csv", row.names = TRUE)

message("Saving the metadata dataframe...")
metadata_df <- spe@metadata$CellSPA$spatialMetricsDiversity$results
write.csv(df, "extdata/metadata_df.csv")

message("Graphing each quantitative metric...")
y_list <- list("pixel_size", "eccentricity", "total_reads", "total_genes", "total_transciprts", 
               "elongation", "compactness", "sphericity", "solidity", "convexity", "circularity", "density",
               "positive_F1", "positive_Precision", "positive_Recall", "negative_F1", "negative_Precision", 
               "negative_Recall", "positive_exprsPct", "negative_exprsPct", "cell_area", "sizeFactor")

for (i in seq_along(y_list)) {
  y_name = y_list[[i]]
  status_message <- glue("Graphing {y_name}...")
  message(status_message)
  p <- scater::plotColData(spe, y_name, 
                           x = "mean_celltype_correlation", 
                           colour_by = "mean_celltype_correlation") +
           theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
  filename <- glue("{data_dir}/{y_name}_plot.pdf")
  ggsave(filename, plot = p, width = 8, height = 8)
}

message("Done!")
