# OPUS Miner R driver.
# Usage: Rscript run_opusminer.R <transactions_file> <out_file> <K>
#
# Transactions file: one line per tx, space-separated 1-based items.
# Output: one itemset per line, space-separated integer items.
suppressPackageStartupMessages({
  library(opusminer)
})

args <- commandArgs(trailingOnly = TRUE)
txn_file <- args[1]
out_file <- args[2]
K        <- as.integer(args[3])

# Read transactions: list of integer vectors
lines <- readLines(txn_file)
transactions <- lapply(lines, function(ln) {
  if (nchar(trimws(ln)) == 0) integer(0)
  else as.integer(strsplit(trimws(ln), "\\s+")[[1]])
})

# Run OPUS Miner: mine top-K productive + self-sufficient itemsets
# by default (alpha = 0.05 for statistical self-sufficiency, search
# space capped at size 5 to match SAME's k_max).
res <- tryCatch({
  opus(transactions,
        k = K,
        format = "data.frame",
        print_closures = FALSE,
        filter_itemsets = TRUE,
        search_by_lift = FALSE,
        correct_for_mult_compare = TRUE,
        redundancy_tests = TRUE)
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n", file = stderr())
  NULL
})

if (is.null(res)) {
  writeLines(character(0), out_file)
  quit(status = 1)
}

# res is a data frame with columns: itemset, count, value, alpha, ...
lines_out <- sapply(res$itemset, function(items) paste(items, collapse = " "))
writeLines(lines_out, out_file)
