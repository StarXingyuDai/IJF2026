rm(list=ls())
library(reshape2)
library(scales)
library(grid)
library(tidyverse)
library(scales)
library(pheatmap)
library(RColorBrewer)
library(ggthemes)
library(ggrepel)
library(xts)
library(gridExtra)
library(dplyr)
library(tidyr)
library(patchwork)

forecast_name=rep(c('MLP','CRSETAR','CCRM','CIAR','ACI','TARI','VC-ACI'),5)
decomposition_name=c(rep('NoDecom',7),rep('BEMD',7),rep('BVMD',7),
                     rep('SVD',7),rep('ISVD',7))
forecast_name2=rep(c('MLP','CRSETAR','CCRM','CIAR','ACI','TARI','VC-ACI'),4)
decomposition_name2=c(rep('BEMD',7),rep('BVMD',7),rep('SVD',7),rep('ISVD',7))
dates=seq.Date(from=as.Date("2000-01-01"),to=as.Date("2026-03-01"),by="month")


####Figure 2####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figure2")

ITS_returns<-read.csv('int_real.csv',header = T,stringsAsFactors = F)

ITS_returns_figure_fun<-function(i,title){
  
  x_breaks <- seq(min(dates), max(dates), length.out = 6)
  
  df_seg <- data.frame(
    date = dates,
    left_endpoint = ITS_returns[,i],
    right_endpoint = ITS_returns[,i+1])
  
  df_shade <- data.frame(
    xmin = as.Date(c('2001-03-01', '2007-12-01', '2020-02-01')),
    xmax = as.Date(c('2001-11-01', '2009-06-01', '2020-04-01'))
  )
  
  ggplot(df_seg) +
    geom_rect(
      data = df_shade,
      aes(
        xmin = xmin,
        xmax = xmax,
        ymin = -Inf,
        ymax = Inf
      ),
      inherit.aes = FALSE,
      fill = "lightgrey",
      alpha = 0.7
    ) +
    geom_line(
      aes(x = date, y = left_endpoint),
      colour = "#1f77b4",
      linewidth = 1) +
    geom_line(
      aes(x = date, y = right_endpoint),
      colour = "#FAAB18",
      linewidth = 1) +
    scale_x_date(breaks = x_breaks,labels = function(x) format(x, "%Y-%m"))+
    theme_bw() +
    theme(
      legend.position = 'none',
      legend.background = element_blank(),
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 14),
      axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5,
                                 face = "bold", size = rel(1.3)),
      axis.text.y = element_text(face = "bold"),
      axis.title.x = element_text(size = 14, face = "bold"),
      axis.title.y = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 14,face = 'bold')) +
    labs(title = title,x = NULL,y = 'ITS returns')%>%
    return()
  
}

p1<-ITS_returns_figure_fun(1,'A) S&P 500 ITS returns')
p2<-ITS_returns_figure_fun(3,'B) WTI ITS returns')
p3<-ITS_returns_figure_fun(5,'C) T-bond ITS returns')
p4<-ITS_returns_figure_fun(7,'D) USDX ITS returns')

grid.newpage()
pushViewport(viewport(layout = grid.layout(4,1))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1))
print(p2, vp = vplayout(2,1))
print(p3, vp = vplayout(3,1))
print(p4, vp = vplayout(4,1))
dev.off()#1000*1000



####Figure 3####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figure3")

BEMD_asset0_dt<-read.csv('decom_asset0_BEMD.csv',header = T,stringsAsFactors = F)
BVMD_asset0_dt<-read.csv('decom_asset0_BVMD.csv',header = T,stringsAsFactors = F)
SVD_asset0_dt<-read.csv('decom_asset0_SVD.csv',header = T,stringsAsFactors = F)
ISVD_asset0_dt<-read.csv('decom_asset0_ISVD.csv',header = T,stringsAsFactors = F)

win_len=84

Raw_figure_fun<-function(dt,title){
  
  x_breaks <- seq(min(dates[1:win_len]), max(dates[1:win_len]), length.out = 6)
  
  df_seg <- data.frame(
    date = dates[1:win_len],
    y = dt[1:win_len,1],
    yend = dt[1:win_len,2])%>%
    mutate(col_group = ifelse(y > yend, "blue", "yellow"),
           lower = pmin(y, yend),
           upper = pmax(y, yend))
  
  p1 <- ggplot(df_seg) +
    geom_segment(
      aes(x = date, xend = date,y = y, yend = yend,colour = col_group),
      alpha = 1,size = 2) +
    scale_colour_manual(
      values = c("blue" = "#1f77b4", "yellow" = "#FAAB18"),
      labels = c("blue" = "Left-endpoints > Right-endpoints",
                 "yellow" = "Left-endpoints < Right-endpoints"))+
    scale_x_date(breaks = x_breaks,labels = function(x) format(x, "%Y-%m"))+
    theme_bw() +
    theme(
      legend.position = 'bottom',
      legend.background = element_blank(),
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 14),
      axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5,
                                 face = "bold", size = rel(1.3)),
      axis.text.y = element_text(face = "bold"),
      axis.title.x = element_text(size = 14, face = "bold"),
      axis.title.y = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 14,face = 'bold')) +
    labs(title = title,x = NULL,y = 'ITS returns',colour = '')
  return(p1)
}
Decompostion_figure_fun<-function(dt,i,title){
  
  x_breaks <- seq(min(dates[1:win_len]), max(dates[1:win_len]), length.out = 6)
  
  df_seg <- data.frame(
    date = dates[1:win_len],
    y = dt[1:win_len,i],
    yend = dt[1:win_len,i+3])%>%
    mutate(col_group = ifelse(y > yend, "blue", "yellow"),
           lower = pmin(y, yend),
           upper = pmax(y, yend))
  
  p1 <- ggplot(df_seg) +
    geom_segment(
      aes(x = date, xend = date,y = y, yend = yend,colour = col_group),
      alpha = 1,size = 2) +
    scale_colour_manual(
      values = c("blue" = "#1f77b4", "yellow" = "#FAAB18"),
      labels = c("blue" = "Left-endpoints > Right-endpoints",
                 "yellow" = "Left-endpoints < Right-endpoints"))+
    scale_x_date(breaks = x_breaks,labels = function(x) format(x, "%Y-%m"))+
    theme_bw() +
    theme(
      legend.position = 'bottom',
      legend.background = element_blank(),
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 14),
      axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5,
                                 face = "bold", size = rel(1.3)),
      axis.text.y = element_text(face = "bold"),
      axis.title.x = element_text(size = 14, face = "bold"),
      axis.title.y = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 14,face = 'bold')) +
    labs(title = title,x = NULL,y = 'ITS returns',colour = '')
  return(p1)
}

p1<-Raw_figure_fun(BEMD_asset0_dt,'A) S&P500 Raw (First 84-month rolling window)')
p2<-Decompostion_figure_fun(BEMD_asset0_dt,5,'B1) S&P500 BEMD Layer 1 (First 84-month rolling window)')
p3<-Decompostion_figure_fun(BEMD_asset0_dt,4,'B2) S&P500 BEMD Layer 2 (First 84-month rolling window)')
p4<-Decompostion_figure_fun(BEMD_asset0_dt,3,'B3) S&P500 BEMD Layer 3 (First 84-month rolling window)')
p5<-Decompostion_figure_fun(BVMD_asset0_dt,5,'C1) S&P500 BVMD Layer 1 (First 84-month rolling window)')
p6<-Decompostion_figure_fun(BVMD_asset0_dt,4,'C2) S&P500 BVMD Layer 2 (First 84-month rolling window)')
p7<-Decompostion_figure_fun(BVMD_asset0_dt,3,'C3) S&P500 BVMD Layer 3 (First 84-month rolling window)')
p8<-Decompostion_figure_fun(SVD_asset0_dt,3,'D1) S&P500 SVD Layer 1 (First 84-month rolling window)')
p9<-Decompostion_figure_fun(SVD_asset0_dt,4,'D2) S&P500 SVD Layer 2 (First 84-month rolling window)')
p10<-Decompostion_figure_fun(SVD_asset0_dt,5,'D3) S&P500 SVD Layer 3 (First 84-month rolling window)')
p11<-Decompostion_figure_fun(ISVD_asset0_dt,3,'E1) S&P500 ISVD Layer 1 (First 84-month rolling window)')
p12<-Decompostion_figure_fun(ISVD_asset0_dt,4,'E2) S&P500 ISVD Layer 2 (First 84-month rolling window)')
p13<-Decompostion_figure_fun(ISVD_asset0_dt,5,'E3) S&P500 ISVD Layer 3 (First 84-month rolling window)')

grid.newpage()
pushViewport(viewport(layout = grid.layout(7,2))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1:2))
print(p2, vp = vplayout(2,1))
print(p3, vp = vplayout(3,1))
print(p4, vp = vplayout(4,1))
print(p5, vp = vplayout(2,2))
print(p6, vp = vplayout(3,2))
print(p7, vp = vplayout(4,2))
print(p8, vp = vplayout(5,1))
print(p9, vp = vplayout(6,1))
print(p10, vp = vplayout(7,1))
print(p11, vp = vplayout(5,2))
print(p12, vp = vplayout(6,2))
print(p13, vp = vplayout(7,2))
dev.off()#1500*2000



####Figure B1####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figureB1")

BEMD_asset1_dt<-read.csv('decom_asset1_BEMD.csv',header = T,stringsAsFactors = F)
BVMD_asset1_dt<-read.csv('decom_asset1_BVMD.csv',header = T,stringsAsFactors = F)
SVD_asset1_dt<-read.csv('decom_asset1_SVD.csv',header = T,stringsAsFactors = F)
ISVD_asset1_dt<-read.csv('decom_asset1_ISVD.csv',header = T,stringsAsFactors = F)

win_len=84

Raw_figure_fun<-function(dt,title){
  
  x_breaks <- seq(min(tail(dates,win_len)),
                  max(tail(dates,win_len)),
                  length.out = 6)
  
  df_seg <- data.frame(
    date = tail(dates,win_len),
    y = dt[1:win_len,1],
    yend = dt[1:win_len,2])%>%
    mutate(col_group = ifelse(y > yend, "blue", "yellow"),
           lower = pmin(y, yend),
           upper = pmax(y, yend))
  
  p1 <- ggplot(df_seg) +
    geom_segment(
      aes(x = date, xend = date,y = y, yend = yend,colour = col_group),
      alpha = 1,size = 2) +
    scale_colour_manual(
      values = c("blue" = "#1f77b4", "yellow" = "#FAAB18"),
      labels = c("blue" = "Left-endpoints > Right-endpoints",
                 "yellow" = "Left-endpoints < Right-endpoints"))+
    scale_x_date(breaks = x_breaks,labels = function(x) format(x, "%Y-%m"))+
    theme_bw() +
    theme(
      legend.position = 'bottom',
      legend.background = element_blank(),
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 14),
      axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5,
                                 face = "bold", size = rel(1.3)),
      axis.text.y = element_text(face = "bold"),
      axis.title.x = element_text(size = 14, face = "bold"),
      axis.title.y = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 14,face = 'bold')) +
    labs(title = title,x = NULL,y = 'ITS returns',colour = '')
  return(p1)
}
Decompostion_figure_fun<-function(dt,i,title){
  
  x_breaks <- seq(min(tail(dates,win_len)),
                  max(tail(dates,win_len)),
                  length.out = 6)
  
  df_seg <- data.frame(
    date = tail(dates,win_len),
    y = dt[1:win_len,i],
    yend = dt[1:win_len,i+3])%>%
    mutate(col_group = ifelse(y > yend, "blue", "yellow"),
           lower = pmin(y, yend),
           upper = pmax(y, yend))
  
  p1 <- ggplot(df_seg) +
    geom_segment(
      aes(x = date, xend = date,y = y, yend = yend,colour = col_group),
      alpha = 1,size = 2) +
    scale_colour_manual(
      values = c("blue" = "#1f77b4", "yellow" = "#FAAB18"),
      labels = c("blue" = "Left-endpoints > Right-endpoints",
                 "yellow" = "Left-endpoints < Right-endpoints"))+
    scale_x_date(breaks = x_breaks,labels = function(x) format(x, "%Y-%m"))+
    theme_bw() +
    theme(
      legend.position = 'bottom',
      legend.background = element_blank(),
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 14),
      axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5,
                                 face = "bold", size = rel(1.3)),
      axis.text.y = element_text(face = "bold"),
      axis.title.x = element_text(size = 14, face = "bold"),
      axis.title.y = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 14,face = 'bold')) +
    labs(title = title,x = NULL,y = 'ITS returns',colour = '')
  return(p1)
}

p1<-Raw_figure_fun(BEMD_asset1_dt,'A) WTI Raw (Last 84-month rolling window)')
p2<-Decompostion_figure_fun(BEMD_asset1_dt,5,'B1) WTI BEMD Layer 1 (Last 84-month rolling window)')
p3<-Decompostion_figure_fun(BEMD_asset1_dt,4,'B2) WTI BEMD Layer 2 (Last 84-month rolling window)')
p4<-Decompostion_figure_fun(BEMD_asset1_dt,3,'B3) WTI BEMD Layer 3 (Last 84-month rolling window)')
p5<-Decompostion_figure_fun(BVMD_asset1_dt,5,'C1) WTI BVMD Layer 1 (Last 84-month rolling window)')
p6<-Decompostion_figure_fun(BVMD_asset1_dt,4,'C2) WTI BVMD Layer 2 (Last 84-month rolling window)')
p7<-Decompostion_figure_fun(BVMD_asset1_dt,3,'C3) WTI BVMD Layer 3 (Last 84-month rolling window)')
p8<-Decompostion_figure_fun(SVD_asset1_dt,3,'D1) WTI SVD Layer 1 (Last 84-month rolling window)')
p9<-Decompostion_figure_fun(SVD_asset1_dt,4,'D2) WTI SVD Layer 2 (Last 84-month rolling window)')
p10<-Decompostion_figure_fun(SVD_asset1_dt,5,'D3) WTI SVD Layer 3 (Last 84-month rolling window)')
p11<-Decompostion_figure_fun(ISVD_asset1_dt,3,'E1) WTI ISVD Layer 1 (Last 84-month rolling window)')
p12<-Decompostion_figure_fun(ISVD_asset1_dt,4,'E2) WTI ISVD Layer 2 (Last 84-month rolling window)')
p13<-Decompostion_figure_fun(ISVD_asset1_dt,5,'E3) WTI ISVD Layer 3 (Last 84-month rolling window)')

grid.newpage()
pushViewport(viewport(layout = grid.layout(7,2))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1:2))
print(p2, vp = vplayout(2,1))
print(p3, vp = vplayout(3,1))
print(p4, vp = vplayout(4,1))
print(p5, vp = vplayout(2,2))
print(p6, vp = vplayout(3,2))
print(p7, vp = vplayout(4,2))
print(p8, vp = vplayout(5,1))
print(p9, vp = vplayout(6,1))
print(p10, vp = vplayout(7,1))
print(p11, vp = vplayout(5,2))
print(p12, vp = vplayout(6,2))
print(p13, vp = vplayout(7,2))
dev.off()#1500*2000


####Figure B2####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figureB2")

BEMD_asset2_dt<-read.csv('decom_asset2_BEMD.csv',header = T,stringsAsFactors = F)
BVMD_asset2_dt<-read.csv('decom_asset2_BVMD.csv',header = T,stringsAsFactors = F)
SVD_asset2_dt<-read.csv('decom_asset2_SVD.csv',header = T,stringsAsFactors = F)
ISVD_asset2_dt<-read.csv('decom_asset2_ISVD.csv',header = T,stringsAsFactors = F)

win_len=84

Raw_figure_fun<-function(dt,title){
  
  x_breaks <- seq(min(tail(dates,win_len)),
                  max(tail(dates,win_len)),
                  length.out = 6)
  
  df_seg <- data.frame(
    date = tail(dates,win_len),
    y = dt[1:win_len,1],
    yend = dt[1:win_len,2])%>%
    mutate(col_group = ifelse(y > yend, "blue", "yellow"),
           lower = pmin(y, yend),
           upper = pmax(y, yend))
  
  p1 <- ggplot(df_seg) +
    geom_segment(
      aes(x = date, xend = date,y = y, yend = yend,colour = col_group),
      alpha = 1,size = 2) +
    scale_colour_manual(
      values = c("blue" = "#1f77b4", "yellow" = "#FAAB18"),
      labels = c("blue" = "Left-endpoints > Right-endpoints",
                 "yellow" = "Left-endpoints < Right-endpoints"))+
    scale_x_date(breaks = x_breaks,labels = function(x) format(x, "%Y-%m"))+
    theme_bw() +
    theme(
      legend.position = 'bottom',
      legend.background = element_blank(),
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 14),
      axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5,
                                 face = "bold", size = rel(1.3)),
      axis.text.y = element_text(face = "bold"),
      axis.title.x = element_text(size = 14, face = "bold"),
      axis.title.y = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 14,face = 'bold')) +
    labs(title = title,x = NULL,y = 'ITS returns',colour = '')
  return(p1)
}
Decompostion_figure_fun<-function(dt,i,title){
  
  x_breaks <- seq(min(tail(dates,win_len)),
                  max(tail(dates,win_len)),
                  length.out = 6)
  
  df_seg <- data.frame(
    date = tail(dates,win_len),
    y = dt[1:win_len,i],
    yend = dt[1:win_len,i+3])%>%
    mutate(col_group = ifelse(y > yend, "blue", "yellow"),
           lower = pmin(y, yend),
           upper = pmax(y, yend))
  
  p1 <- ggplot(df_seg) +
    geom_segment(
      aes(x = date, xend = date,y = y, yend = yend,colour = col_group),
      alpha = 1,size = 2) +
    scale_colour_manual(
      values = c("blue" = "#1f77b4", "yellow" = "#FAAB18"),
      labels = c("blue" = "Left-endpoints > Right-endpoints",
                 "yellow" = "Left-endpoints < Right-endpoints"))+
    scale_x_date(breaks = x_breaks,labels = function(x) format(x, "%Y-%m"))+
    theme_bw() +
    theme(
      legend.position = 'bottom',
      legend.background = element_blank(),
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 14),
      axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5,
                                 face = "bold", size = rel(1.3)),
      axis.text.y = element_text(face = "bold"),
      axis.title.x = element_text(size = 14, face = "bold"),
      axis.title.y = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 14,face = 'bold')) +
    labs(title = title,x = NULL,y = 'ITS returns',colour = '')
  return(p1)
}

p1<-Raw_figure_fun(BEMD_asset2_dt,'A) T-bond Raw (Last 84-month rolling window)')
p2<-Decompostion_figure_fun(BEMD_asset2_dt,5,'B1) T-bond BEMD Layer 1 (Last 84-month rolling window)')
p3<-Decompostion_figure_fun(BEMD_asset2_dt,4,'B2) T-bond BEMD Layer 2 (Last 84-month rolling window)')
p4<-Decompostion_figure_fun(BEMD_asset2_dt,3,'B3) T-bond BEMD Layer 3 (Last 84-month rolling window)')
p5<-Decompostion_figure_fun(BVMD_asset2_dt,5,'C1) T-bond BVMD Layer 1 (Last 84-month rolling window)')
p6<-Decompostion_figure_fun(BVMD_asset2_dt,4,'C2) T-bond BVMD Layer 2 (Last 84-month rolling window)')
p7<-Decompostion_figure_fun(BVMD_asset2_dt,3,'C3) T-bond BVMD Layer 3 (Last 84-month rolling window)')
p8<-Decompostion_figure_fun(SVD_asset2_dt,3,'D1) T-bond SVD Layer 1 (Last 84-month rolling window)')
p9<-Decompostion_figure_fun(SVD_asset2_dt,4,'D2) T-bond SVD Layer 2 (Last 84-month rolling window)')
p10<-Decompostion_figure_fun(SVD_asset2_dt,5,'D3) T-bond SVD Layer 3 (Last 84-month rolling window)')
p11<-Decompostion_figure_fun(ISVD_asset2_dt,3,'E1) T-bond ISVD Layer 1 (Last 84-month rolling window)')
p12<-Decompostion_figure_fun(ISVD_asset2_dt,4,'E2) T-bond ISVD Layer 2 (Last 84-month rolling window)')
p13<-Decompostion_figure_fun(ISVD_asset2_dt,5,'E3) T-bond ISVD Layer 3 (Last 84-month rolling window)')

grid.newpage()
pushViewport(viewport(layout = grid.layout(7,2))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1:2))
print(p2, vp = vplayout(2,1))
print(p3, vp = vplayout(3,1))
print(p4, vp = vplayout(4,1))
print(p5, vp = vplayout(2,2))
print(p6, vp = vplayout(3,2))
print(p7, vp = vplayout(4,2))
print(p8, vp = vplayout(5,1))
print(p9, vp = vplayout(6,1))
print(p10, vp = vplayout(7,1))
print(p11, vp = vplayout(5,2))
print(p12, vp = vplayout(6,2))
print(p13, vp = vplayout(7,2))
dev.off()#1500*2000




####Figure B3####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figureB3")

BEMD_asset3_dt<-read.csv('decom_asset3_BEMD.csv',header = T,stringsAsFactors = F)
BVMD_asset3_dt<-read.csv('decom_asset3_BVMD.csv',header = T,stringsAsFactors = F)
SVD_asset3_dt<-read.csv('decom_asset3_SVD.csv',header = T,stringsAsFactors = F)
ISVD_asset3_dt<-read.csv('decom_asset3_ISVD.csv',header = T,stringsAsFactors = F)

win_len=84

Raw_figure_fun<-function(dt,title){
  
  x_breaks <- seq(min(head(dates,win_len)),
                  max(head(dates,win_len)),
                  length.out = 6)
  
  df_seg <- data.frame(
    date = head(dates,win_len),
    y = dt[1:win_len,1],
    yend = dt[1:win_len,2])%>%
    mutate(col_group = ifelse(y > yend, "blue", "yellow"),
           lower = pmin(y, yend),
           upper = pmax(y, yend))
  
  p1 <- ggplot(df_seg) +
    geom_segment(
      aes(x = date, xend = date,y = y, yend = yend,colour = col_group),
      alpha = 1,size = 2) +
    scale_colour_manual(
      values = c("blue" = "#1f77b4", "yellow" = "#FAAB18"),
      labels = c("blue" = "Left-endpoints > Right-endpoints",
                 "yellow" = "Left-endpoints < Right-endpoints"))+
    scale_x_date(breaks = x_breaks,labels = function(x) format(x, "%Y-%m"))+
    theme_bw() +
    theme(
      legend.position = 'bottom',
      legend.background = element_blank(),
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 14),
      axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5,
                                 face = "bold", size = rel(1.3)),
      axis.text.y = element_text(face = "bold"),
      axis.title.x = element_text(size = 14, face = "bold"),
      axis.title.y = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 14,face = 'bold')) +
    labs(title = title,x = NULL,y = 'ITS returns',colour = '')
  return(p1)
}
Decompostion_figure_fun<-function(dt,i,title){
  
  x_breaks <- seq(min(head(dates,win_len)),
                  max(head(dates,win_len)),
                  length.out = 6)
  
  df_seg <- data.frame(
    date = head(dates,win_len),
    y = dt[1:win_len,i],
    yend = dt[1:win_len,i+3])%>%
    mutate(col_group = ifelse(y > yend, "blue", "yellow"),
           lower = pmin(y, yend),
           upper = pmax(y, yend))
  
  p1 <- ggplot(df_seg) +
    geom_segment(
      aes(x = date, xend = date,y = y, yend = yend,colour = col_group),
      alpha = 1,size = 2) +
    scale_colour_manual(
      values = c("blue" = "#1f77b4", "yellow" = "#FAAB18"),
      labels = c("blue" = "Left-endpoints > Right-endpoints",
                 "yellow" = "Left-endpoints < Right-endpoints"))+
    scale_x_date(breaks = x_breaks,labels = function(x) format(x, "%Y-%m"))+
    theme_bw() +
    theme(
      legend.position = 'bottom',
      legend.background = element_blank(),
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 14),
      axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5,
                                 face = "bold", size = rel(1.3)),
      axis.text.y = element_text(face = "bold"),
      axis.title.x = element_text(size = 14, face = "bold"),
      axis.title.y = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 14,face = 'bold')) +
    labs(title = title,x = NULL,y = 'ITS returns',colour = '')
  return(p1)
}

p1<-Raw_figure_fun(BEMD_asset3_dt,'A) USDX Raw (First 84-month rolling window)')
p2<-Decompostion_figure_fun(BEMD_asset3_dt,5,'B1) USDX BEMD Layer 1 (First 84-month rolling window)')
p3<-Decompostion_figure_fun(BEMD_asset3_dt,4,'B2) USDX BEMD Layer 2 (First 84-month rolling window)')
p4<-Decompostion_figure_fun(BEMD_asset3_dt,3,'B3) USDX BEMD Layer 3 (First 84-month rolling window)')
p5<-Decompostion_figure_fun(BVMD_asset3_dt,5,'C1) USDX BVMD Layer 1 (First 84-month rolling window)')
p6<-Decompostion_figure_fun(BVMD_asset3_dt,4,'C2) USDX BVMD Layer 2 (First 84-month rolling window)')
p7<-Decompostion_figure_fun(BVMD_asset3_dt,3,'C3) USDX BVMD Layer 3 (First 84-month rolling window)')
p8<-Decompostion_figure_fun(SVD_asset3_dt,3,'D1) USDX SVD Layer 1 (First 84-month rolling window)')
p9<-Decompostion_figure_fun(SVD_asset3_dt,4,'D2) USDX SVD Layer 2 (First 84-month rolling window)')
p10<-Decompostion_figure_fun(SVD_asset3_dt,5,'D3) USDX SVD Layer 3 (First 84-month rolling window)')
p11<-Decompostion_figure_fun(ISVD_asset3_dt,3,'E1) USDX ISVD Layer 1 (First 84-month rolling window)')
p12<-Decompostion_figure_fun(ISVD_asset3_dt,4,'E2) USDX ISVD Layer 2 (First 84-month rolling window)')
p13<-Decompostion_figure_fun(ISVD_asset3_dt,5,'E3) USDX ISVD Layer 3 (First 84-month rolling window)')

grid.newpage()
pushViewport(viewport(layout = grid.layout(7,2))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1:2))
print(p2, vp = vplayout(2,1))
print(p3, vp = vplayout(3,1))
print(p4, vp = vplayout(4,1))
print(p5, vp = vplayout(2,2))
print(p6, vp = vplayout(3,2))
print(p7, vp = vplayout(4,2))
print(p8, vp = vplayout(5,1))
print(p9, vp = vplayout(6,1))
print(p10, vp = vplayout(7,1))
print(p11, vp = vplayout(5,2))
print(p12, vp = vplayout(6,2))
print(p13, vp = vplayout(7,2))
dev.off()#1500*2000





####Figure 4####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figure4")

raw_asset0_period_dt<-read.csv('raw_asset0_period.csv',header = T,stringsAsFactors = F)
BEMD_asset0_period_dt<-read.csv('BEMD_asset0_period.csv',header = T,stringsAsFactors = F)
BVMD_asset0_period_dt<-read.csv('BVMD_asset0_period.csv',header = T,stringsAsFactors = F)
SVD_asset0_period_dt<-read.csv('SVD_asset0_period.csv',header = T,stringsAsFactors = F)
ISVD_asset0_period_dt<-read.csv('ISVD_asset0_period.csv',header = T,stringsAsFactors = F)

Periodogram_figure_fun <- function(dt, title){
  xx_value <- (1:nrow(raw_asset0_period_dt)) / nrow(raw_asset0_period_dt) / 2
  
  ggplot() +
    geom_line(
      aes(x = xx_value, y = raw_asset0_period_dt$X0, colour = "Raw"),
      size = 7, alpha = 0.7
    ) +
    geom_line(
      aes(x = xx_value, y = dt$X0, colour = "Layer 1"),
      size = 3
    ) +
    geom_line(
      aes(x = xx_value, y = dt$X1, colour = "Layer 2"),
      size = 3
    ) +
    geom_line(
      aes(x = xx_value, y = dt$X2, colour = "Layer 3"),
      size = 3
    ) +
    scale_colour_manual(
      values = c(
        "Raw" = "grey",
        "Layer 1" = "#FAAB18",
        "Layer 2" = "#1380A1",
        "Layer 3" = "#990000"
      ),
      name = NULL
    ) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.background = element_blank(),
      axis.text.x = element_text(angle = 0, hjust = 1, face = "bold", size = rel(1.3)),
      axis.text.y = element_text(face = "bold")
    ) +
    labs(
      title = title,
      x = "Frequency (Hz)",
      y = "Amplitude"
    )
}

p1<-Periodogram_figure_fun(BEMD_asset0_period_dt,
                       'A) Periodogram of WTI ITS sub-returns decomposed by BEMD')
p2<-Periodogram_figure_fun(BVMD_asset0_period_dt,
                       'B) Periodogram of WTI ITS sub-returns decomposed by BVMD')
p3<-Periodogram_figure_fun(SVD_asset0_period_dt,
                       'C) Periodogram of WTI ITS sub-returns decomposed by SVD')
p4<-Periodogram_figure_fun(ISVD_asset0_period_dt,
                       'D) Periodogram of WTI ITS sub-returns decomposed by ISVD')

grid.newpage()
pushViewport(viewport(layout = grid.layout(2,2))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1))
print(p2, vp = vplayout(1,2))
print(p3, vp = vplayout(2,1))
print(p4, vp = vplayout(2,2))
dev.off()#1200*1000


####Figure B4####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figureB4")

raw_asset0_period_dt<-read.csv('raw_asset0_period.csv',header = T,stringsAsFactors = F)
BEMD_asset0_period_dt<-read.csv('BEMD_asset0_period.csv',header = T,stringsAsFactors = F)
BVMD_asset0_period_dt<-read.csv('BVMD_asset0_period.csv',header = T,stringsAsFactors = F)
SVD_asset0_period_dt<-read.csv('SVD_asset0_period.csv',header = T,stringsAsFactors = F)
ISVD_asset0_period_dt<-read.csv('ISVD_asset0_period.csv',header = T,stringsAsFactors = F)

Periodogram_figure_fun <- function(dt, title){
  xx_value <- (1:nrow(raw_asset0_period_dt)) / nrow(raw_asset0_period_dt) / 2
  
  ggplot() +
    geom_line(
      aes(x = xx_value, y = raw_asset0_period_dt$X0, colour = "Raw"),
      size = 7, alpha = 0.7
    ) +
    geom_line(
      aes(x = xx_value, y = dt$X0, colour = "Layer 1"),
      size = 3
    ) +
    geom_line(
      aes(x = xx_value, y = dt$X1, colour = "Layer 2"),
      size = 3
    ) +
    geom_line(
      aes(x = xx_value, y = dt$X2, colour = "Layer 3"),
      size = 3
    ) +
    scale_colour_manual(
      values = c(
        "Raw" = "grey",
        "Layer 1" = "#FAAB18",
        "Layer 2" = "#1380A1",
        "Layer 3" = "#990000"
      ),
      name = NULL
    ) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.background = element_blank(),
      axis.text.x = element_text(angle = 0, hjust = 1, face = "bold", size = rel(1.3)),
      axis.text.y = element_text(face = "bold")
    ) +
    labs(
      title = title,
      x = "Frequency (Hz)",
      y = "Amplitude"
    )
}

p1<-Periodogram_figure_fun(BEMD_asset0_period_dt,
                           'A) Periodogram of S&P 500 ITS sub-returns decomposed by BEMD')
p2<-Periodogram_figure_fun(BVMD_asset0_period_dt,
                           'B) Periodogram of S&P 500 ITS sub-returns decomposed by BVMD')
p3<-Periodogram_figure_fun(SVD_asset0_period_dt,
                           'C) Periodogram of S&P 500 ITS sub-returns decomposed by SVD')
p4<-Periodogram_figure_fun(ISVD_asset0_period_dt,
                           'D) Periodogram of S&P 500 ITS sub-returns decomposed by ISVD')

grid.newpage()
pushViewport(viewport(layout = grid.layout(2,2))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1))
print(p2, vp = vplayout(1,2))
print(p3, vp = vplayout(2,1))
print(p4, vp = vplayout(2,2))
dev.off()#1200*1000


####Figure B5####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figureB5")

raw_asset0_period_dt<-read.csv('raw_asset0_period.csv',header = T,stringsAsFactors = F)
BEMD_asset0_period_dt<-read.csv('BEMD_asset0_period.csv',header = T,stringsAsFactors = F)
BVMD_asset0_period_dt<-read.csv('BVMD_asset0_period.csv',header = T,stringsAsFactors = F)
SVD_asset0_period_dt<-read.csv('SVD_asset0_period.csv',header = T,stringsAsFactors = F)
ISVD_asset0_period_dt<-read.csv('ISVD_asset0_period.csv',header = T,stringsAsFactors = F)

Periodogram_figure_fun <- function(dt, title){
  xx_value <- (1:nrow(raw_asset0_period_dt)) / nrow(raw_asset0_period_dt) / 2
  
  ggplot() +
    geom_line(
      aes(x = xx_value, y = raw_asset0_period_dt$X0, colour = "Raw"),
      size = 7, alpha = 0.7
    ) +
    geom_line(
      aes(x = xx_value, y = dt$X0, colour = "Layer 1"),
      size = 3
    ) +
    geom_line(
      aes(x = xx_value, y = dt$X1, colour = "Layer 2"),
      size = 3
    ) +
    geom_line(
      aes(x = xx_value, y = dt$X2, colour = "Layer 3"),
      size = 3
    ) +
    scale_colour_manual(
      values = c(
        "Raw" = "grey",
        "Layer 1" = "#FAAB18",
        "Layer 2" = "#1380A1",
        "Layer 3" = "#990000"
      ),
      name = NULL
    ) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.background = element_blank(),
      axis.text.x = element_text(angle = 0, hjust = 1, face = "bold", size = rel(1.3)),
      axis.text.y = element_text(face = "bold")
    ) +
    labs(
      title = title,
      x = "Frequency (Hz)",
      y = "Amplitude"
    )
}

p1<-Periodogram_figure_fun(BEMD_asset0_period_dt,
                           'A) Periodogram of T-bond ITS sub-returns decomposed by BEMD')
p2<-Periodogram_figure_fun(BVMD_asset0_period_dt,
                           'B) Periodogram of T-bond ITS sub-returns decomposed by BVMD')
p3<-Periodogram_figure_fun(SVD_asset0_period_dt,
                           'C) Periodogram of T-bond ITS sub-returns decomposed by SVD')
p4<-Periodogram_figure_fun(ISVD_asset0_period_dt,
                           'D) Periodogram of T-bond ITS sub-returns decomposed by ISVD')

grid.newpage()
pushViewport(viewport(layout = grid.layout(2,2))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1))
print(p2, vp = vplayout(1,2))
print(p3, vp = vplayout(2,1))
print(p4, vp = vplayout(2,2))
dev.off()#1200*1000



####Figure B6####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figureB6")

raw_asset0_period_dt<-read.csv('raw_asset0_period.csv',header = T,stringsAsFactors = F)
BEMD_asset0_period_dt<-read.csv('BEMD_asset0_period.csv',header = T,stringsAsFactors = F)
BVMD_asset0_period_dt<-read.csv('BVMD_asset0_period.csv',header = T,stringsAsFactors = F)
SVD_asset0_period_dt<-read.csv('SVD_asset0_period.csv',header = T,stringsAsFactors = F)
ISVD_asset0_period_dt<-read.csv('ISVD_asset0_period.csv',header = T,stringsAsFactors = F)

Periodogram_figure_fun <- function(dt, title){
  xx_value <- (1:nrow(raw_asset0_period_dt)) / nrow(raw_asset0_period_dt) / 2
  
  ggplot() +
    geom_line(
      aes(x = xx_value, y = raw_asset0_period_dt$X0, colour = "Raw"),
      size = 7, alpha = 0.7
    ) +
    geom_line(
      aes(x = xx_value, y = dt$X0, colour = "Layer 1"),
      size = 3
    ) +
    geom_line(
      aes(x = xx_value, y = dt$X1, colour = "Layer 2"),
      size = 3
    ) +
    geom_line(
      aes(x = xx_value, y = dt$X2, colour = "Layer 3"),
      size = 3
    ) +
    scale_colour_manual(
      values = c(
        "Raw" = "grey",
        "Layer 1" = "#FAAB18",
        "Layer 2" = "#1380A1",
        "Layer 3" = "#990000"
      ),
      name = NULL
    ) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.background = element_blank(),
      axis.text.x = element_text(angle = 0, hjust = 1, face = "bold", size = rel(1.3)),
      axis.text.y = element_text(face = "bold")
    ) +
    labs(
      title = title,
      x = "Frequency (Hz)",
      y = "Amplitude"
    )
}

p1<-Periodogram_figure_fun(BEMD_asset0_period_dt,
                           'A) Periodogram of USDX ITS sub-returns decomposed by BEMD')
p2<-Periodogram_figure_fun(BVMD_asset0_period_dt,
                           'B) Periodogram of USDX ITS sub-returns decomposed by BVMD')
p3<-Periodogram_figure_fun(SVD_asset0_period_dt,
                           'C) Periodogram of USDX ITS sub-returns decomposed by SVD')
p4<-Periodogram_figure_fun(ISVD_asset0_period_dt,
                           'D) Periodogram of USDX ITS sub-returns decomposed by ISVD')

grid.newpage()
pushViewport(viewport(layout = grid.layout(2,2))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1))
print(p2, vp = vplayout(1,2))
print(p3, vp = vplayout(2,1))
print(p4, vp = vplayout(2,2))
dev.off()#1200*1000


####Figure 5####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figure5")
All_asset0_dt<-read.csv('All_asset0_w84.csv',header = T,stringsAsFactors = F)
All_asset1_dt<-read.csv('All_asset1_w84.csv',header = T,stringsAsFactors = F)
All_asset2_dt<-read.csv('All_asset2_w84.csv',header = T,stringsAsFactors = F)
All_asset3_dt<-read.csv('All_asset3_w84.csv',header = T,stringsAsFactors = F)

All_asset0_dt2<-cbind(All_asset0_dt$MDE_mr%>%rank%>%floor(),
                      All_asset0_dt$MDE_lr%>%rank%>%floor())
All_asset0_dt2<-All_asset0_dt2[-c(1:7),]
colnames(All_asset0_dt2)<-c('x1','x2')

All_asset1_dt2<-cbind(All_asset1_dt$MDE_mr%>%rank%>%floor(),
                      All_asset1_dt$MDE_lr%>%rank%>%floor())
All_asset1_dt2<-All_asset1_dt2[-c(1:7),]
colnames(All_asset1_dt2)<-c('x1','x2')

All_asset2_dt2<-cbind(All_asset2_dt$MDE_mr%>%rank%>%floor(),
                      All_asset2_dt$MDE_lr%>%rank%>%floor())
All_asset2_dt2<-All_asset2_dt2[-c(1:7),]
colnames(All_asset2_dt2)<-c('x1','x2')

All_asset3_dt2<-cbind(All_asset3_dt$MDE_mr%>%rank%>%floor(),
                      All_asset3_dt$MDE_lr%>%rank%>%floor())
All_asset3_dt2<-All_asset3_dt2[-c(1:7),]
colnames(All_asset3_dt2)<-c('x1','x2')

loss_tidy_fun<-function(dt){ 
  dt<-dt%>% as_tibble()%>% 
    mutate(forecast=forecast_name2, 
           decomposition=decomposition_name2)
  table1<-dt%>% group_by(forecast)%>% 
    summarise(`D_cr average rank`=mean(x1),
              `D_LR average rank`=mean(x2)) 
  table2<-dt%>% group_by(decomposition)%>%
    summarise(`D_cr average rank`=mean(x1),
              `D_LR average rank`=mean(x2))
  return(list(table1,table2))
}
forecast_foregroup_figure_fun<-function(dt,title){
  loss_tidy_fun(dt)[[1]]%>% 
    melt(id='forecast')%>% 
    mutate(value=value%>%round(3))%>%
    ggplot(aes(x=forecast,y=value,fill=forecast))+
    facet_wrap(.~ variable, scales='free_y',nrow = 1)+
    geom_bar(stat='identity',position = 'dodge')+
    scale_fill_manual(values=c("#1f77b4","#ff7f0e","#2ca02c", "#d62728","#9467bd","#8c564b","#e377c2"), 
                      labels=c('MLP','CRSETAR','CCRM','CIAR','ACI','TARI','VC-ACI'))+ 
    geom_text(aes(label=value),vjust=-0.2,position = position_dodge(1),size=3.5)+ 
    theme_bw()+ 
    theme(legend.position='none',
          axis.text.x=element_text(face = "bold",angle = 28,
                                   vjust=0.5,hjust = 1,margin = margin(t = -4)),
          axis.text.y=element_text(face = "bold"))+
    labs(title=title, 
         x='',y='', 
         fill='')%>%
    return()
}
forecast_decomgroup_figure_fun<-function(dt,title){
  loss_tidy_fun(dt)[[2]]%>% 
    melt(id='decomposition')%>% 
    mutate(value=value%>%round(3))%>%
    ggplot(aes(x=decomposition,y=value,fill=decomposition))+
    facet_wrap(.~ variable, scales='free_y',nrow = 1)+
    geom_bar(stat='identity',position = 'dodge')+
    scale_fill_manual(values=c("#1f77b4","#ff7f0e","#2ca02c", "#d62728","#9467bd"), 
                      labels=c('NoDecom','BEMD','BVMD','SVD','ISVD'))+ 
    geom_text(aes(label=value),vjust=-0.2,position = position_dodge(1),size=3.5)+ 
    theme_bw()+ 
    theme(legend.position='none',
          axis.text.x=element_text(face = "bold",angle = 28,
                                   vjust=0.5,hjust = 1,margin = margin(t = -4)),
          axis.text.y=element_text(face = "bold"))+
    labs(title=title, 
         x='',y='', 
         fill='')
}

p1<-forecast_foregroup_figure_fun(All_asset0_dt2,'A) S&P500 ITS in different forecast approach groups')
p2<-forecast_decomgroup_figure_fun(All_asset0_dt2,'B) S&P500 ITS in different decomposition algorithm groups')
p3<-forecast_foregroup_figure_fun(All_asset1_dt2,'C) WTI ITS in different forecast approach groups')
p4<-forecast_decomgroup_figure_fun(All_asset1_dt2,'D) WTI ITS in different decomposition algorithm groups')
p5<-forecast_foregroup_figure_fun(All_asset2_dt2,'E) T-bonds ITS in different forecast approach groups')
p6<-forecast_decomgroup_figure_fun(All_asset2_dt2,'F) T-bonds ITS in different decomposition algorithm groups')
p7<-forecast_foregroup_figure_fun(All_asset3_dt2,'G) USDX ITS in different forecast approach groups')
p8<-forecast_decomgroup_figure_fun(All_asset3_dt2,'H) USDX ITS in different decomposition algorithm groups')

grid.newpage()
pushViewport(viewport(layout = grid.layout(4,2))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1))
print(p2, vp = vplayout(1,2))
print(p3, vp = vplayout(2,1))
print(p4, vp = vplayout(2,2))
print(p5, vp = vplayout(3,1))
print(p6, vp = vplayout(3,2))
print(p7, vp = vplayout(4,1))
print(p8, vp = vplayout(4,2))
dev.off()#1500*1600



####Figure 6####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figure6")

cols_11_ssci <- c("#1B4F72","#A93226","#117A65","#7D3C98",  
  "#B9770E","#566573","#2E86C1","#CB4335","#239B56","#884EA0","#CA6F1E")

cumloss_asset0<-read.csv('cumloss_asset0.csv',header = T,stringsAsFactors = F)
cumloss_asset1<-read.csv('cumloss_asset1.csv',header = T,stringsAsFactors = F)
cumloss_asset2<-read.csv('cumloss_asset2.csv',header = T,stringsAsFactors = F)
cumloss_asset3<-read.csv('cumloss_asset3.csv',header = T,stringsAsFactors = F)

cumloss_figure_fun<-function(dt,i,title,y_title){
  
  df_shade <- data.frame(
    xmin = as.Date(c('2007-12-01', '2020-02-01')),
    xmax = as.Date(c('2009-06-01', '2020-04-01'))
  )
  
  df0 <- tibble(
    date=as.Date(dates[-c(1:84)]),
    `MLP`=dt[,i+4*0],
    `CRSETAR`=dt[,i+4*1],
    `CCRM`=dt[,i+4*2],
    `CIAR`=dt[,i+4*3],
    `ACI`=dt[,i+4*4],
    `TARI`=dt[,i+4*5],
    `VC-ACI`=dt[,i+4*6],
    `BEMD-VC-ACI`=dt[,i+4*7],
    `BVMD-VC-ACI`=dt[,i+4*8],
    `SVD-VC-ACI`=dt[,i+4*9],
    `ISVD-VC-ACI`=dt[,i+4*10]
  ) %>%
    melt(id='date')
  
  df_end <- df0 %>%
    group_by(variable) %>%
    filter(!is.na(value)) %>%
    slice_tail(n=1) %>%
    ungroup()
  
  x_breaks <- seq(min(df0$date), max(df0$date), length.out = 6)
  
  ggplot(df0, aes(x=date,y=value,color=variable,
                  fill=variable,size=variable,linetype=variable)) +
    geom_rect(
      data = df_shade,
      aes(
        xmin = xmin,
        xmax = xmax,
        ymin = -Inf,
        ymax = Inf
      ),
      inherit.aes = FALSE,
      fill = "lightgrey",
      alpha = 0.7
    )+
    geom_line(alpha=1)+
    geom_text_repel(
      data = df_end,
      aes(label = variable),
      direction = "y",
      hjust = 0,
      #nudge_x = 20,
      nudge_x = 5,
      segment.color = NA,
      show.legend = FALSE,
      size = 4,
      fontface = "bold"
    )+
    scale_fill_manual(values=cols_11_ssci)+
    scale_color_manual(values=cols_11_ssci)+
    scale_size_manual(values = c(rep(1,11)))+
    scale_linetype_manual(values = c(rep(1,11)))+
    scale_x_date(
      breaks = x_breaks,
      labels = function(x) format(x, "%Y-%m"),
      expand = expansion(mult = c(0.02, 0.18))
    )+
    coord_cartesian(clip = "off")+
    theme_bw()+
    theme(
      legend.position='none',
      axis.text.x=element_text(angle = 0,hjust = 0.5,face = "bold",size = rel(1.3)),
      axis.text.y=element_text(face = "bold"),
      #plot.margin = margin(5.5, 80, 5.5, 5.5)
      plot.margin = margin(3, 6, 3, 3)
    )+
    labs(title=title,
         x='Date',y=y_title)%>%
    return()
}

p1<-cumloss_figure_fun(cumloss_asset0,2,'A) S&P500 Left/Right-endpoint performance compared with the VC-ACI model',
                       'Cumulative RMSE')
p2<-cumloss_figure_fun(cumloss_asset0,4,'B) S&P500 Left/Right-endpoint performance compared with the Random Walk',
                       'Cumulative RMSE')
p3<-cumloss_figure_fun(cumloss_asset1,2,'C) WTI Left/Right-endpoint performance compared with the VC-ACI model',
                       'Cumulative RMSE')
p4<-cumloss_figure_fun(cumloss_asset1,4,'D) WTI Left/Right-endpoint performance compared with the Random Walk',
                       'Cumulative RMSE')
p5<-cumloss_figure_fun(cumloss_asset2,2,'E) T-bond Left/Right-endpoint performance compared with the VC-ACI model',
                       'Cumulative RMSE')
p6<-cumloss_figure_fun(cumloss_asset2,4,'F) T-bond Left/Right-endpoint performance compared with the Random Walk',
                       'Cumulative RMSE')
p7<-cumloss_figure_fun(cumloss_asset3,2,'G) USDX Left/Right-endpoint performance compared with the VC-ACI model',
                       'Cumulative RMSE')
p8<-cumloss_figure_fun(cumloss_asset3,4,'H) USDX Left/Right-endpoint performance compared with the Random Walk',
                       'cumulative RMSE')

fig6 <- (p1 + p2) /
  (p3 + p4) /
  (p5 + p6) /
  (p7 + p8)

ggsave(
  filename = "Figure 6.png",
  plot = fig6,
  width = 18*0.9,
  height = 22*0.9,
  dpi = 200,
  bg = "white"
)


####Figure 7####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figure7")
cvr0_dt<-read.csv('cvr_all_asset0.csv',header = T,stringsAsFactors = F)*100
cvr1_dt<-read.csv('cvr_all_asset1.csv',header = T,stringsAsFactors = F)*100
cvr2_dt<-read.csv('cvr_all_asset2.csv',header = T,stringsAsFactors = F)*100
cvr3_dt<-read.csv('cvr_all_asset3.csv',header = T,stringsAsFactors = F)*100


cvr_tidy_fun<-function(dt){ 
  dt<-dt%>% tibble()%>% 
    mutate(forecast=forecast_name, 
           decomposition=decomposition_name)
  table1<-dt%>% group_by(forecast)%>% 
    summarise(`Coverage rate`=mean(Coverage)%>%round(2),
              `Efficiency rate`=mean(Efficiency)%>%round(2)) 
  table2<-dt%>% group_by(decomposition)%>%
    summarise(`Coverage rate`=mean(Coverage)%>%round(2),
              `Efficiency rate`=mean(Efficiency)%>%round(2))
  return(list(table1,table2)) 
} 

cvr_foregroup_figure_fun<-function(dt,title){
  cvr_tidy_fun(dt)[[1]]%>% 
    melt(id='forecast')%>% 
    mutate(value=value%>%round(3))%>%
    ggplot(aes(x=forecast,y=value,fill=forecast))+
    facet_wrap(.~ variable, scales='free_y',nrow = 1)+
    geom_bar(stat='identity',position = 'dodge')+
    scale_fill_manual(values=c("#1f77b4","#ff7f0e","#2ca02c", "#d62728","#9467bd","#8c564b","#e377c2"), 
                      labels=c('MLP','CRSETAR','CCRM','CIAR','ACI','TARI','VC-ACI'))+ 
    geom_text(aes(label=value),vjust=-0.2,position = position_dodge(1),size=3.5)+ 
    theme_bw()+ 
    theme(legend.position='none',
          axis.text.x=element_text(face = "bold",angle = 35,
                                   vjust=0.5,hjust = 1,margin = margin(t = -4)),
          axis.text.y=element_text(face = "bold"))+
    labs(title=title, 
         x='',y='%', 
         fill='')%>%
    return()
}

cvr_decomgroup_figure_fun<-function(dt,title){
  cvr_tidy_fun(dt)[[2]]%>% 
    melt(id='decomposition')%>% 
    mutate(value=value%>%round(3))%>%
    ggplot(aes(x=decomposition,y=value,fill=decomposition))+
    facet_wrap(.~ variable, scales='free_y',nrow = 1)+
    geom_bar(stat='identity',position = 'dodge')+
    scale_fill_manual(values=c("#1f77b4","#ff7f0e","#2ca02c", "#d62728","#9467bd"), 
                      labels=c('NoDecom','BEMD','BVMD','SVD','ISVD'))+ 
    geom_text(aes(label=value),vjust=-0.2,position = position_dodge(1),size=3.5)+ 
    theme_bw()+ 
    theme(legend.position='none',
          axis.text.x=element_text(face = "bold",angle = 35,
                                   vjust=0.5,hjust = 1,margin = margin(t = -4)),
          axis.text.y=element_text(face = "bold"))+
    labs(title=title, 
         x='',y='%', 
         fill='')
}


p1<-cvr_foregroup_figure_fun(cvr0_dt,'A) S&P500 ITS in different forecast approach groups')
p2<-cvr_decomgroup_figure_fun(cvr0_dt,'B) S&P500 ITS in different decomposition algorithm groups')
p3<-cvr_foregroup_figure_fun(cvr1_dt,'C) WTI ITS in different forecast approach groups')
p4<-cvr_decomgroup_figure_fun(cvr1_dt,'D) WTI ITS in different decomposition algorithm groups')
p5<-cvr_foregroup_figure_fun(cvr2_dt,'E) T-bond ITS in different forecast approach groups')
p6<-cvr_decomgroup_figure_fun(cvr2_dt,'F) T-bond ITS in different decomposition algorithm groups')
p7<-cvr_foregroup_figure_fun(cvr3_dt,'G) USDX ITS in different forecast approach groups')
p8<-cvr_decomgroup_figure_fun(cvr3_dt,'H) USDX ITS in different decomposition algorithm groups')


grid.newpage()
pushViewport(viewport(layout = grid.layout(4,2))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1))
print(p2, vp = vplayout(1,2))
print(p3, vp = vplayout(2,1))
print(p4, vp = vplayout(2,2))
print(p5, vp = vplayout(3,1))
print(p6, vp = vplayout(3,2))
print(p7, vp = vplayout(4,1))
print(p8, vp = vplayout(4,2))
dev.off()#1500*1600



####Figure 8####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figure8")
multih_cr_asset0<-read.csv('multih_p4_w84_l3_crloss_asset0.csv',header = T,stringsAsFactors = F)
multih_cr_asset1<-read.csv('multih_p4_w84_l3_crloss_asset1.csv',header = T,stringsAsFactors = F)
multih_cr_asset2<-read.csv('multih_p4_w84_l3_crloss_asset2.csv',header = T,stringsAsFactors = F)
multih_cr_asset3<-read.csv('multih_p4_w84_l3_crloss_asset3.csv',header = T,stringsAsFactors = F)
multih_lr_asset0<-read.csv('multih_p4_w84_l3_lrloss_asset0.csv',header = T,stringsAsFactors = F)
multih_lr_asset1<-read.csv('multih_p4_w84_l3_lrloss_asset1.csv',header = T,stringsAsFactors = F)
multih_lr_asset2<-read.csv('multih_p4_w84_l3_lrloss_asset2.csv',header = T,stringsAsFactors = F)
multih_lr_asset3<-read.csv('multih_p4_w84_l3_lrloss_asset3.csv',header = T,stringsAsFactors = F)

multih_figure_fun <- function(dt, title) {
  
  df_long <- dt %>%
    rename(model = X) %>%
    pivot_longer(
      cols = starts_with("h"),
      names_to = "horizon",
      values_to = "rank"
    ) %>%
    mutate(
      h_num = as.numeric(sub("h", "", horizon)),
      x_pos = match(h_num, c(2, 3, 5))
    )
  
  df_end <- df_long %>%
    group_by(model) %>%
    filter(h_num == max(h_num)) %>%
    ungroup()
  
  p <- ggplot(
    df_long,
    aes(x = x_pos, y = rank, color = model, group = model)
  ) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2.2) +
    geom_text(
      data = df_end,
      aes(x = x_pos + 0.12, label = model),
      hjust = 0,
      size = 3.5,
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_x_continuous(
      breaks = c(1, 2, 3),
      labels = c(2, 3, 5),
      expand = expansion(mult = c(0.02, 0.25))
    ) +
    scale_y_reverse(
      breaks = 1:11
    ) +
    coord_cartesian(clip = "off") +
    theme_bw() +
    theme(
      legend.position = "none",
      axis.text.x = element_text(face = "bold", size = 12),
      axis.text.y = element_text(face = "bold", size = 12),
      axis.title.x = element_text(face = "bold", size = 12),
      axis.title.y = element_text(face = "bold", size = 12),
      plot.title = element_text(face = "bold", size = 13),
      plot.margin = margin(5.5, 100, 5.5, 5.5),
      
      panel.border = element_blank(),
      axis.line.x.bottom = element_line(color = "black"),
      axis.line.y.left = element_line(color = "black"),
      axis.line.x.top = element_blank(),
      axis.line.y.right = element_blank()
    ) +
    labs(
      title = title,
      x = "Forecast horizon",
      y = "Ranking"
    )
  
  return(p)
}

p11<-multih_figure_fun(multih_cr_asset0,
                  expression(paste("A1) S&P 500 ranking for ", D[cr], ' (84-month)')))
p21<-multih_figure_fun(multih_cr_asset1,
                      expression(paste("B1) WTI ranking for ", D[cr], ' (84-month)')))
p31<-multih_figure_fun(multih_cr_asset2,
                      expression(paste("C1) T-bond ranking for ", D[cr], ' (84-month)')))
p41<-multih_figure_fun(multih_cr_asset3,
                      expression(paste("D1) USDX ranking for ", D[cr], ' (84-month)')))

p12<-multih_figure_fun(multih_lr_asset0,
                      expression(paste("A2) S&P 500 ranking for ", D[LR], ' (84-month)')))
p22<-multih_figure_fun(multih_lr_asset1,
                      expression(paste("B2) WTI ranking for ", D[LR], ' (84-month)')))
p32<-multih_figure_fun(multih_lr_asset2,
                      expression(paste("C2) T-bond ranking for ", D[LR], ' (84-month)')))
p42<-multih_figure_fun(multih_lr_asset3,
                      expression(paste("D2) USDX ranking for ", D[LR], ' (84-month)')))

multih_cr_asset0<-read.csv('multih_p4_w96_l3_crloss_asset0.csv',header = T,stringsAsFactors = F)
multih_cr_asset1<-read.csv('multih_p4_w96_l3_crloss_asset1.csv',header = T,stringsAsFactors = F)
multih_cr_asset2<-read.csv('multih_p4_w96_l3_crloss_asset2.csv',header = T,stringsAsFactors = F)
multih_cr_asset3<-read.csv('multih_p4_w96_l3_crloss_asset3.csv',header = T,stringsAsFactors = F)
multih_lr_asset0<-read.csv('multih_p4_w96_l3_lrloss_asset0.csv',header = T,stringsAsFactors = F)
multih_lr_asset1<-read.csv('multih_p4_w96_l3_lrloss_asset1.csv',header = T,stringsAsFactors = F)
multih_lr_asset2<-read.csv('multih_p4_w96_l3_lrloss_asset2.csv',header = T,stringsAsFactors = F)
multih_lr_asset3<-read.csv('multih_p4_w96_l3_lrloss_asset3.csv',header = T,stringsAsFactors = F)


p13<-multih_figure_fun(multih_cr_asset0,
                      expression(paste("E1) S&P 500 ranking for ", D[cr], ' (96-month)')))
p23<-multih_figure_fun(multih_cr_asset1,
                      expression(paste("F1) WTI ranking for ", D[cr], ' (96-month)')))
p33<-multih_figure_fun(multih_cr_asset2,
                      expression(paste("G1) T-bond ranking for ", D[cr], ' (96-month)')))
p43<-multih_figure_fun(multih_cr_asset3,
                      expression(paste("H1) USDX ranking for ", D[cr], ' (96-month)')))

p14<-multih_figure_fun(multih_lr_asset0,
                      expression(paste("E2) S&P 500 ranking for ", D[LR], ' (96-month)')))
p24<-multih_figure_fun(multih_lr_asset1,
                      expression(paste("F2) WTI ranking for ", D[LR], ' (96-month)')))
p34<-multih_figure_fun(multih_lr_asset2,
                      expression(paste("G2) T-bond ranking for ", D[LR], ' (96-month)')))
p44<-multih_figure_fun(multih_lr_asset3,
                      expression(paste("H2) USDX ranking for ", D[LR], ' (96-month)')))

grid.newpage()
pushViewport(viewport(layout = grid.layout(4,4))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p11, vp = vplayout(1,1))
print(p12, vp = vplayout(1,2))
print(p13, vp = vplayout(1,3))
print(p14, vp = vplayout(1,4))
print(p21, vp = vplayout(2,1))
print(p22, vp = vplayout(2,2))
print(p23, vp = vplayout(2,3))
print(p24, vp = vplayout(2,4))
print(p31, vp = vplayout(3,1))
print(p32, vp = vplayout(3,2))
print(p33, vp = vplayout(3,3))
print(p34, vp = vplayout(3,4))
print(p41, vp = vplayout(4,1))
print(p42, vp = vplayout(4,2))
print(p43, vp = vplayout(4,3))
print(p44, vp = vplayout(4,4))
dev.off()#1500*1200





####Figure 9####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figure9")
multip_cr_asset0<-read.csv('multip_f1_w84_l3_crloss_asset0.csv',header = T,stringsAsFactors = F)
multip_lr_asset0<-read.csv('multip_f1_w84_l3_lrloss_asset0.csv',header = T,stringsAsFactors = F)
multip_cr_asset1<-read.csv('multip_f1_w84_l3_crloss_asset1.csv',header = T,stringsAsFactors = F)
multip_lr_asset1<-read.csv('multip_f1_w84_l3_lrloss_asset1.csv',header = T,stringsAsFactors = F)
multip_cr_asset2<-read.csv('multip_f1_w84_l3_crloss_asset2.csv',header = T,stringsAsFactors = F)
multip_lr_asset2<-read.csv('multip_f1_w84_l3_lrloss_asset2.csv',header = T,stringsAsFactors = F)
multip_cr_asset3<-read.csv('multip_f1_w84_l3_crloss_asset3.csv',header = T,stringsAsFactors = F)
multip_lr_asset3<-read.csv('multip_f1_w84_l3_lrloss_asset3.csv',header = T,stringsAsFactors = F)


multip_figure_fun<-function(dt,title){
  df_long <- dt %>%
    rename(model = X) %>%
    pivot_longer(
      cols = starts_with("p"),
      names_to = 'Lag',
      values_to = "rank"
    ) %>%
    mutate(
      h_num = as.numeric(sub("p", "", Lag))
    )
  
  df_long$h_num[which(df_long$h_num==1)]=1
  df_long$h_num[which(df_long$h_num==2)]=2
  df_long$h_num[which(df_long$h_num==3)]=3
  df_long$h_num[which(df_long$h_num==5)]=4
  
  df_end <- df_long %>%
    group_by(model) %>%
    filter(h_num == max(h_num)) %>%
    ungroup()
  
  p <- ggplot(df_long, aes(x = h_num, y = rank, color = model, group = model)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2.2) +
    geom_text(
      data = df_end,
      aes(x = h_num + 0.35, label = model),
      hjust = 0,
      size = 3.5,
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_x_continuous(
      breaks = 1:4,
      labels = c('P=1','P=2','P=3','P=5'),
      expand = expansion(mult = c(0.02, 0.25))
    )  +
    scale_y_reverse(
      breaks = 1:11
    ) +
    coord_cartesian(clip = "off") +
    theme_bw() +
    theme(
      legend.position = "none",
      axis.text.x = element_text(face = "bold", size = 12),
      axis.text.y = element_text(face = "bold", size = 12),
      axis.title.x = element_text(face = "bold", size = 12),
      axis.title.y = element_text(face = "bold", size = 12),
      plot.title = element_text(face = "bold", size = 13),
      plot.margin = margin(5.5, 100, 5.5, 5.5)
    ) +
    labs(
      title = title,
      x = "Lag order",
      y = "Rank"
    )
  return(p)
}

p1<-multip_figure_fun(multip_cr_asset0,
                      expression(paste("A) S&P 500 ranking for ", D[cr])))
p3<-multip_figure_fun(multip_cr_asset1,
                      expression(paste("C) WTI ranking for ", D[cr])))
p5<-multip_figure_fun(multip_cr_asset2,
                      expression(paste("E) T-bond ranking for ", D[cr])))
p7<-multip_figure_fun(multip_cr_asset3,
                      expression(paste("G) USDX ranking for ", D[cr])))

p2<-multip_figure_fun(multip_lr_asset0,
                      expression(paste("B) S&P 500 ranking for ", D[LR])))
p4<-multip_figure_fun(multip_lr_asset1,
                      expression(paste("D) WTI ranking for ", D[LR])))
p6<-multip_figure_fun(multip_lr_asset2,
                      expression(paste("F) T-bond ranking for ", D[LR])))
p8<-multip_figure_fun(multip_lr_asset3,
                      expression(paste("H) USDX ranking for ", D[LR])))

grid.newpage()
pushViewport(viewport(layout = grid.layout(4,2))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1))
print(p2, vp = vplayout(1,2))
print(p3, vp = vplayout(2,1))
print(p4, vp = vplayout(2,2))
print(p5, vp = vplayout(3,1))
print(p6, vp = vplayout(3,2))
print(p7, vp = vplayout(4,1))
print(p8, vp = vplayout(4,2))
dev.off()#1400*1200



####Figure 10####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figure10")
vol_loss0_dt<-read.csv('vol_loss_0.csv',header = T,stringsAsFactors = F)
vol_loss1_dt<-read.csv('vol_loss_1.csv',header = T,stringsAsFactors = F)
vol_loss2_dt<-read.csv('vol_loss_2.csv',header = T,stringsAsFactors = F)
vol_loss3_dt<-read.csv('vol_loss_3.csv',header = T,stringsAsFactors = F)

vol_loss0_rank_dt=cbind(vol_loss0_dt$Qlike%>%rank%>%floor(),
                        vol_loss0_dt$MSE%>%rank%>%floor())
vol_loss1_rank_dt=cbind(vol_loss1_dt$Qlike%>%rank%>%floor(),
                        vol_loss1_dt$MSE%>%rank%>%floor())
vol_loss2_rank_dt=cbind(vol_loss2_dt$Qlike%>%rank%>%floor(),
                        vol_loss2_dt$MSE%>%rank%>%floor())
vol_loss3_rank_dt=cbind(vol_loss3_dt$Qlike%>%rank%>%floor(),
                        vol_loss3_dt$MSE%>%rank%>%floor())

colnames(vol_loss0_rank_dt)<-c('Qlike_rank','MSE_rank')
colnames(vol_loss1_rank_dt)<-c('Qlike_rank','MSE_rank')
colnames(vol_loss2_rank_dt)<-c('Qlike_rank','MSE_rank')
colnames(vol_loss3_rank_dt)<-c('Qlike_rank','MSE_rank')

dt=vol_loss0_rank_dt
title=''
  
vloss_tidy_fun<-function(dt){ 
  dt<-dt%>% as_tibble()%>% 
    mutate(forecast=forecast_name2, 
           decomposition=decomposition_name2)
  table1<-dt%>% group_by(forecast)%>% 
    summarise(`Qlike's average rank`=mean(Qlike_rank),
              `MSE's average rank`=mean(MSE_rank)) 
  table2<-dt%>% group_by(decomposition)%>%
    summarise(`Qlike's average rank`=mean(Qlike_rank),
              `MSE's average rank`=mean(MSE_rank))
  return(list(table1,table2)) 
}

vforecast_foregroup_figure_fun<-function(dt,title){
  vloss_tidy_fun(dt)[[1]]%>% 
    melt(id='forecast')%>% 
    mutate(value=value%>%round(3))%>%
    ggplot(aes(x=forecast,y=value,fill=forecast))+
    facet_wrap(.~ variable, scales='free_y',nrow = 1)+
    geom_bar(stat='identity',position = 'dodge')+
    scale_fill_manual(values=c("#1f77b4","#ff7f0e","#2ca02c", "#d62728","#9467bd","#8c564b","#e377c2"), 
                      labels=c('MLP','CRSETAR','CCRM','CIAR','ACI','TARI','VC-ACI'))+ 
    geom_text(aes(label=value),vjust=-0.5,position = position_dodge(1),size=3.5)+ 
    theme_bw()+ 
    theme(legend.position='none',
          axis.text.x=element_text(face = "bold",angle = 35,
                                   vjust=0.5,hjust = 1,margin = margin(t = -4)),
          axis.text.y=element_text(face = "bold"))+
    labs(title=title, 
         x=NULL,y=NULL, 
         fill='')%>%
    return()
}

vforecast_decomgroup_figure_fun<-function(dt,title){
  vloss_tidy_fun(dt)[[2]]%>% 
    melt(id='decomposition')%>% 
    mutate(value=value%>%round(3))%>%
    ggplot(aes(x=decomposition,y=value,fill=decomposition))+
    facet_wrap(.~ variable, scales='free_y',nrow = 1)+
    geom_bar(stat='identity',position = 'dodge')+
    scale_fill_manual(values=c("#1f77b4","#ff7f0e","#2ca02c", "#d62728","#9467bd"), 
                      labels=c('NoDecom','BEMD','BVMD','SVD','ISVD'))+ 
    geom_text(aes(label=value),vjust=-0.5,position = position_dodge(1),size=3.5)+ 
    theme_bw()+ 
    theme(legend.position='none',
          axis.text.x=element_text(face = "bold",angle = 35,
                                   vjust=0.5,hjust = 1,margin = margin(t = -4)),
          axis.text.y=element_text(face = "bold"))+
    labs(title=title, 
         x=NULL,y=NULL, 
         fill='')
}

p1<-vforecast_foregroup_figure_fun(vol_loss0_rank_dt,'A) S&P500 in forecast approach groups')
p2<-vforecast_decomgroup_figure_fun(vol_loss0_rank_dt,'B) S&P500 in decomposition algorithm groups')
p3<-vforecast_foregroup_figure_fun(vol_loss1_rank_dt,'C) WTI variance in forecast approach groups')
p4<-vforecast_decomgroup_figure_fun(vol_loss1_rank_dt,'D) WTI variance in decomposition algorithm groups')
p5<-vforecast_foregroup_figure_fun(vol_loss2_rank_dt,'E) T-bond variance in forecast approach groups')
p6<-vforecast_decomgroup_figure_fun(vol_loss2_rank_dt,'F) T-bond variance in decomposition algorithm groups')
p7<-vforecast_foregroup_figure_fun(vol_loss3_rank_dt,'G) USDX variance in forecast approach groups')
p8<-vforecast_decomgroup_figure_fun(vol_loss3_rank_dt,'H) USDX variance in decomposition algorithm groups')

grid.newpage()
pushViewport(viewport(layout = grid.layout(4,2))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1))
print(p2, vp = vplayout(1,2))
print(p3, vp = vplayout(2,1))
print(p4, vp = vplayout(2,2))
print(p5, vp = vplayout(3,1))
print(p6, vp = vplayout(3,2))
print(p7, vp = vplayout(4,1))
print(p8, vp = vplayout(4,2))
dev.off()#1500*1700


####Figure 11 ####
setwd("E:/当前论文工作/IJF/IJF提交5/data & code/figures/figure11")
pf_MV_dt<-read.csv('pf_MV.csv',header = T,stringsAsFactors = F)
pf_MV_right_dt<-read.csv('pf_MV_right.csv',header = T,stringsAsFactors = F)
pf_MV_left_dt<-read.csv('pf_MV_left.csv',header = T,stringsAsFactors = F)
pf_GMV_dt<-read.csv('pf_GMV.csv',header = T,stringsAsFactors = F)

pf_MV_S_dt<-read.csv('pf_MV_S.csv',header = T,stringsAsFactors = F)
pf_MV_right_S_dt<-read.csv('pf_MV_right_S.csv',header = T,stringsAsFactors = F)
pf_MV_left_S_dt<-read.csv('pf_MV_left_S.csv',header = T,stringsAsFactors = F)
pf_GMV_S_dt<-read.csv('pf_GMV_S.csv',header = T,stringsAsFactors = F)

pf_MV_dt$MCS<-(-pf_MV_dt$MCS)%>%rank%>%floor()
pf_MV_right_dt$MCS<-(-pf_MV_right_dt$MCS)%>%rank%>%floor()
pf_MV_left_dt$MCS<-(-pf_MV_left_dt$MCS)%>%rank%>%floor()
pf_GMV_dt$MCS<-(-pf_GMV_dt$MCS)%>%rank%>%floor()
pf_MV_S_dt$MCS<-(-pf_MV_S_dt$MCS)%>%rank%>%floor()
pf_MV_right_S_dt$MCS<-(-pf_MV_right_S_dt$MCS)%>%rank%>%floor()
pf_MV_left_S_dt$MCS<-(-pf_MV_left_S_dt$MCS)%>%rank%>%floor()
pf_GMV_S_dt$MCS<-(-pf_GMV_S_dt$MCS)%>%rank%>%floor()


pf_tidy_fun<-function(dt){ 
  dt<-dt%>% tibble()%>% 
    mutate(forecast=forecast_name2, 
           decomposition=decomposition_name2)
  table1<-dt%>% group_by(forecast)%>% 
    summarise(`performance's average rank`=mean(MCS)) 
  table2<-dt%>% group_by(decomposition)%>%
    summarise(`performance's average rank`=mean(MCS)) 
  return(list(table1,table2)) 
}

pf_foregroup_figure_fun<-function(dt,title){
  pf_tidy_fun(dt)[[1]]%>% 
    melt(id='forecast')%>% 
    mutate(value=value%>%round(3))%>%
    ggplot(aes(x=forecast,y=value,fill=forecast))+
    facet_wrap(.~ variable, scales='free_y',nrow = 1)+
    geom_bar(stat='identity',position = 'dodge')+
    scale_fill_manual(values=c("#1f77b4","#ff7f0e","#2ca02c", "#d62728","#9467bd","#8c564b","#e377c2"), 
                      labels=c('MLP','CRSETAR','CCRM','CIAR','ACI','TARI','VC-ACI'))+ 
    geom_text(aes(label=value),vjust=-0.5,position = position_dodge(1),size=3.5)+ 
    theme_bw()+ 
    theme(legend.position='none',
          axis.text.x=element_text(face = "bold",angle = 35,
                                   vjust=0.5,hjust = 1,margin = margin(t = -4)),
          axis.text.y=element_text(face = "bold"))+
    labs(title=title, 
         x='',y='MCS\'s P-value', 
         fill='')%>%
    return()
}

pf_decomgroup_figure_fun<-function(dt,title){
  pf_tidy_fun(dt)[[2]]%>% 
    melt(id='decomposition')%>% 
    mutate(value=value%>%round(3))%>%
    ggplot(aes(x=decomposition,y=value,fill=decomposition))+
    facet_wrap(.~ variable, scales='free_y',nrow = 1)+
    geom_bar(stat='identity',position = 'dodge')+
    scale_fill_manual(values=c("#1f77b4","#ff7f0e","#2ca02c", "#d62728","#9467bd"), 
                      labels=c('NoDecom','BEMD','BVMD','SVD','ISVD'))+ 
    geom_text(aes(label=value),vjust=-0.5,position = position_dodge(1),size=3.5)+ 
    theme_bw()+ 
    theme(legend.position='none',
          axis.text.x=element_text(face = "bold",angle = 35,
                                   vjust=0.5,hjust = 1,margin = margin(t = -4)),
          axis.text.y=element_text(face = "bold"))+
    labs(title=title, 
         x='',y='', 
         fill='')
}

p1<-pf_foregroup_figure_fun(pf_MV_dt,'A) MV portfolio (\U03B1=0.5, 0% short constrant) grouped by forecast approaches')
p2<-pf_decomgroup_figure_fun(pf_MV_dt,'B) MV portfolio (\U03B1=0.5, 0% short constrant) grouped by decomposition algorithm')
p3<-pf_foregroup_figure_fun(pf_MV_S_dt,'C) MV portfolio (\U03B1=0.5, 50% short constrant) grouped by forecast approaches')
p4<-pf_decomgroup_figure_fun(pf_MV_S_dt,'D) MV portfolio (\U03B1=0.5, 50% short constrant) grouped by decomposition algorithm')

p5<-pf_foregroup_figure_fun(pf_MV_right_dt,'E) MV portfolio (\U03B1=0.2, 0% short constrant) grouped by forecast approaches')
p6<-pf_decomgroup_figure_fun(pf_MV_right_dt,'F) MV portfolio (\U03B1=0.2, 0% short constrant) grouped by decomposition algorithm')
p7<-pf_foregroup_figure_fun(pf_MV_right_S_dt,'G) MV portfolio (\U03B1=0.2, 50% short constrant) grouped by forecast approaches')
p8<-pf_decomgroup_figure_fun(pf_MV_right_S_dt,'H) MV portfolio (\U03B1=0.2, 50% short constrant) grouped by decomposition algorithm')

p9<-pf_foregroup_figure_fun(pf_MV_left_dt,'I) MV portfolio (\U03B1=0.8, 0% short constrant) grouped by forecast approaches')
p10<-pf_decomgroup_figure_fun(pf_MV_left_dt,'J) MV portfolio (\U03B1=0.8, 0% short constrant) grouped by decomposition algorithm')
p11<-pf_foregroup_figure_fun(pf_MV_left_S_dt,'K) MV portfolio (\U03B1=0.8, 50% short constrant) grouped by forecast approaches')
p12<-pf_decomgroup_figure_fun(pf_MV_left_S_dt,'L) MV portfolio (\U03B1=0.8, 50% short constrant) grouped by decomposition algorithm')

p13<-pf_foregroup_figure_fun(pf_GMV_dt,'M) GMV portfolio (0% short constrant) grouped by forecast approaches')
p14<-pf_decomgroup_figure_fun(pf_GMV_dt,'N) GMV portfolio (0% short constrant) grouped by decomposition algorithm')
p15<-pf_foregroup_figure_fun(pf_GMV_S_dt,'O) GMV portfolio (50% short constrant) grouped by forecast approaches')
p16<-pf_decomgroup_figure_fun(pf_GMV_S_dt,'P) GMV portfolio (50% short constrant) grouped by decomposition algorithm')

grid.newpage()
pushViewport(viewport(layout = grid.layout(4,4))) 
vplayout <- function(x,y){
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(p1, vp = vplayout(1,1))
print(p2, vp = vplayout(1,2))
print(p3, vp = vplayout(1,3))
print(p4, vp = vplayout(1,4))
print(p5, vp = vplayout(2,1))
print(p6, vp = vplayout(2,2))
print(p7, vp = vplayout(2,3))
print(p8, vp = vplayout(2,4))
print(p9, vp = vplayout(3,1))
print(p10, vp = vplayout(3,2))
print(p11, vp = vplayout(3,3))
print(p12, vp = vplayout(3,4))
print(p13, vp = vplayout(4,1))
print(p14, vp = vplayout(4,2))
print(p15, vp = vplayout(4,3))
print(p16, vp = vplayout(4,4))

dev.off()#3200*1700




