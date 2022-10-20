library(fwildclusterboot)

df <- read.csv("data/test_df.csv")
fit <- lm(Y ~ X1 + X2, df)

tboot_list <- 
lapply(c("11", "31"), function(x){
  lapply(c(TRUE, FALSE), function(y){
    boottest(
      fit, 
      param = "X1",
      clustid = ~cluster,
      B = 99999, 
      bootstrap_type = x, 
      impose_null = y, 
      ssc = boot_ssc(adj = FALSE, cluster.adj = FALSE)
    )$t_boot
  })
})
unlist(tboot_list)
df <- 
  cbind(
    Reduce("cbind",tboot_list[[1]]),
    Reduce("cbind",tboot_list[[2]])
) |> as.data.frame()
names(df) <- c("WCR11", "WCR31", "WCU11", "WCU31")
write.csv(df, "data/test_df_fwc_res.csv")
