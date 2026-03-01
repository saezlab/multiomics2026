#!/usr/bin/env Rscript
#
# Extract TF and kinase activity data from the paper's RData files
# and combine them into a single activities.tsv.
#
# The EV3 xlsx table (MOESM6) only contains TF activities. Kinase
# activities are only available in the RData files from the paper repo.
# This script extracts both from RData and saves a combined file.
#
# Input RData files (from paper-repo/data/):
#   - 2024-07-24_tf_enrichment_results.RData
#   - 2024-07-24_kinase_enrichment_result.RData
#
# Output:
#   - data/differential/activities.tsv
#     Columns: enzyme_type, source, time, score, p_value

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
    script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg)))
} else {
    script_dir <- getwd()
}
rdata_dir <- file.path(dirname(script_dir), "..", "project-prep", "paper-repo", "data")
out_file <- file.path(script_dir, "..", "data", "differential", "activities.tsv")

cat("Loading TF enrichment data...\n")
load(file.path(rdata_dir, "2024-07-24_tf_enrichment_results.RData"))
tf <- res_tf_enrichment$enrichment
tf_out <- data.frame(
    enzyme_type = "TF",
    source = tf$source,
    time = tf$time,
    score = tf$score,
    p_value = tf$p_value
)
cat(sprintf("  %d TF activities (%d TFs x %d time points)\n",
    nrow(tf_out), length(unique(tf_out$source)), length(unique(tf_out$time))))

cat("Loading kinase enrichment data...\n")
load(file.path(rdata_dir, "2024-07-24_kinase_enrichment_result.RData"))
kin <- res_kinase_enrichment$enrichment
# The kinase data has 3 statistics; the paper uses norm_wmean
# (verified by matching scores to the paper's network input object)
kin <- kin[kin$statistic == "norm_wmean", ]
kin_out <- data.frame(
    enzyme_type = "Kinase",
    source = kin$source,
    time = kin$time,
    score = kin$score,
    p_value = kin$p_value
)
cat(sprintf("  %d kinase activities (%d kinases x %d time points, norm_wmean)\n",
    nrow(kin_out), length(unique(kin_out$source)), length(unique(kin_out$time))))

combined <- rbind(tf_out, kin_out)
cat(sprintf("\nCombined: %d rows\n", nrow(combined)))
print(table(combined$enzyme_type))

dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)
write.table(combined, out_file, sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("\nSaved to %s\n", out_file))
