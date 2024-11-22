#import all libraries
library(gmodels)
library(caret)
library(tidyverse)
library(leaps)
library(ggplot2)
library(reshape2)
library(MASS)
library(ggcorrplot)
library(corrplot)
library(plotmo)
library(keras)
library(kableExtra)
library(modelr)
library(psych)
library(Rmisc)
library(gridExtra)
library(scales)
library(rpart)
library(yardstick)
library(cluster)
library(NbClust)
library(factoextra)
library(readxl)

#importing the dataset in the excel file
whiteDat <- read_excel("D:\\IIT\Year 2\\SEMESTER 2\\Machine Learning and Data Mining\\5DATA001C.2\\Coursework\\Machine Learning Coursework Code RStudio\\MachineLearningCWTask1RStudio\\Whitewine_v2.xlsx")

sortByCorr = function(dataset,refColName){
  # Sort the dataframe columns by the absolute value of their correlation with
  # a given column
  #
  # Args:
  #   dataset: A vector, matrix, or data frame to sort
  #   refColName: The name of the reference colum for the correlation
  #
  # Returns:
  #   The sorted dataframe
  refColIdx = grep(refColName,colNames(dataset))
  corrTmp = cor(dataset)[,refColIdx]
  corrTmp[order(abs(corrTmp),decreasing = TRUE)]
  
  dataset[,order(abs(corrTmp), decreasing = TRUE)]
}

#summary
dim(whiteDat)
numericVars <- which(sapply(whiteDat,is.numeric))
numericVarNames <-names(numericVars)
cat("There are",length(numericVars),"numeric variables")

sapply(whiteDat,class)

#taking a look at the data distribution:
head(whiteDat)

#getting the metrics about the variables
summary(whiteDat)

#values which contains outliers
oldpar = par(mfrow = c(2,6))
for(i in 1:11){
  boxplot(whiteDat[[i]])
  mtext(names(whiteDat)[i], cex = 0.8, side = 1 , line = 2 )
}
par(oldpar)

#getting the outliers locations
pairs(whiteDat[,-grep("Quality",colNames(whiteDat))])

#predictor values distribution
oldpar = par(mforw = c(2,6))
for(i in 1:12){
  truehist(whiteDat[[i]], xlab = names(whiteDat)[i], col = 'lightgreen',
           main = paste("Average = ", signif(mean(whiteDat[[i]]),3)))
}
par(oldpar)

#outliers detection
outliers = c()
for(i in 1:11){
  stats = boxplot.stats(whiteDat[[i]])$stats
  bottom_outlier_rows = which(whiteDat[[i]] < stats[1])
  top_outlier_rows = which(whiteDat[[i]] > stats[5])
  outliers = c(outliers , top_outlier_rows[ !top_outlier_rows %in% outliers ] )
  outliers = c(outliers , bottom_outlier_rows[ !bottom_outlier_rows %in% outliers ] )
}

#https://stats.stackexchange.com/questions/164099/removing-outliers-based-on-cooks-distance-in-r-language

mod = lm(quality ~ ., data = whiteDat)
cooksd = cooks.distance(mod)
plot(cooksd, pch = "*", cex = 2, main = "Influential Obs by Cooks distance")
abline(h = 4*mean(cooksd, na.rm = T), col = "red")

head(whiteDat[cooksd > 4 * mean(cooksd, na.rm=T), ])

#remove all the outliers in the list from the dataset and create a new set  of histogram data
coutliers = as.numeric(rownames(whiteDat[cooksd > 4 * mean(cooksd, na.rm=T), ]))
outliers = c(outliers , coutliers[ !coutliers %in% outliers ] )

cleanWhiteDat = whiteDat[-outliers, ]
oldpar = par(mfrow=c(2,6))
for ( i in 1:12 ) {
  truehist(cleanWhiteDat[[i]], xlab = names(cleanWhiteDat)[i], col = 'lightgreen', 
           main = paste("Average =", signif(mean(cleanWhiteDat[[i]]),3)))
}

par(oldpar)
dim(cleanWhiteDat)

#To evaluate if there is a linear association between our variables, we used scatterplot matrice.
pairs(cleanWhiteDat, col = cleanWhiteDat$quality, pch = cleanWhiteDat$quality)

pairs(cleanWhiteDat[,c(7, 8, 10, 11)], col = cleanWhiteDat$quality, pch = cleanWhiteDat$quality)

#correlation matrix
ggcorrplot(cor(cleanWhiteDat), hc.order = TRUE, type = "lower", lab = TRUE, insig = "blank")

colnames(sortByCorr(dataset = cleanWhiteDat, refColName = 'quality'))

#Creating the corrplot
numericVars <- which(sapply(cleanWhiteDat, is.numeric))
all_numVar <- cleanWhiteDat[, numericVars]
cor_numVar <- cor(all_numVar, use = "pairwise.complete.obs")

#Sort on decreasing correlations with alcohol
cor_sorted <- as.matrix(sort(cor_numVar[,"alcohol"], decreasing = TRUE))

#Selecting high correlations 
Cor_High <- names(which(apply(cor_sorted, 1, function(x) abs(x) > 0.175)))
cor_numVar <- cor_numVar[Cor_High, Cor_High]
corrplot.mixed(cor_numVar, tl.col = "black", tl.pos = "lt")

#Data PreProcessing
#Converting the data of all columns to numeric
#Making adjustments to convert data types
#Changing two integer columns to numbers
cleanWhiteDat$quality <- as.numeric(cleanWhiteDat$quality)

#to make it easier to create a data collection without a color column
wineData_no_color <- cleanWhiteDat[1:12]
#check the structure again
str(wineData_no_color)

#Look for Near Zero Variance.
#Will give us a column number, which may have a small variation.
nzv <- nearZeroVar(cleanWhiteDat)
print(paste("***Column number with***", nzv))

#Normalizing the Data i.e Scaling and Centering
#Identifying huge data columns
head(cleanWhiteDat)

#We may deduce from the statistics that Columns 1, 4, 6, 7, 9, 11, and 12 may cause problems, 
#We'll try to normalize them such that their values fall between 0 and 1..
print("---Normalizing Data----")
norm_data <- sapply(cleanWhiteDat[,c(1,4,6,7,9,11,12)], function(x) (x - min(x))/(max(x) - min(x)))
print("---Type of returned data")
class(norm_data)
print("---Converting data from matrix to data.frame---")
norm_data <- data.frame(norm_data)    # norm_data is a 'matrix'
print("---Normalised data---")
head(norm_data)

#Connecting the normalized data to additional information
wineData_norm <- cbind(cleanWhiteDat[,c(2,3,5,8,10)],norm_data)
head(wineData_norm)
str(wineData_norm)
wineData_scaled <- scale(wineData_no_color)
head(wineData_scaled)
class(wineData_scaled)

#Converting to data.frame
wineData_scaled_df <- as.data.frame(wineData_scaled)
class(wineData_scaled_df)

#Identifying highly correlated columns
#Lets find correlation using cor()
corr_norm <- round(cor(wineData_norm),1)
corr_norm

#https://www.rdocumentation.org/packages/ggcorrplot/versions/0.1.3/topics/ggcorrplot
#graphical display of a correlation matrix using ggplot2.
ggcorrplot(corr_norm, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("tomato2", "white", "springgreen3"), 
           title="Correlogram of Wine Data", 
           ggtheme=theme_dark)

corr_scaled <- round(cor(wineData_scaled_df),1)

#graphical display for Scaled data
ggcorrplot(corr_scaled, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("tomato2", "white", "springgreen3"), 
           title="Correlogram of Wine Data", 
           ggtheme=theme_dark)

#Calculating the number of clusters
#Method: 1 - Elbow method
# Initialize total within sum of squares error: wss
wss <- 0

# For 1 to 20 cluster centers
for (i in 1:10) {
  km.out <- kmeans(wineData_norm, centers = i)
  # Save the sum of squares total to the wss variable.
  wss[i] <- km.out$tot.withinss
}
wss

# Plot the amount of squares overall vs the number of clusters
plot(1:10, wss, type = "b", 
     xlab = "Number of Clusters", 
     ylab = "Within groups sum of squares")

#Method: 2 Silhouette plot
wine_train_data <- sample_frac(wineData_norm, 0.65)
head(wine_train_data)
str(wine_train_data)
nrow(wine_train_data)

#plot NbCluster-based silhouette plot
fviz_nbclust(wine_train_data, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")

#Segmentation Using K-mean
km <- kmeans(wineData_norm, 2, iter.max = 140 , algorithm="Lloyd", nstart=100)
km

#Structure of km
str(km)

#graph view
# Centroid Plot against 1st 2nd discriminant functions
clusplot(cleanWhiteDat, km$cluster, color=TRUE, shade=TRUE, 
         labels=2, lines=0)

fviz_cluster(km, data = wineData_norm,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal())

fviz_cluster(list(data = wineData_norm, cluster = km$cluster),
             ellipse.type = "norm", geom = "point", stand = FALSE,
             palette = "jco", ggtheme = theme_classic())

pam.res <- pam(wineData_norm, 2)

# Visualize
fviz_cluster(pam.res)

