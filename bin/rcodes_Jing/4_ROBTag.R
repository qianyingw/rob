#-----load library
# install.packages("githubinstall")
# library(githubinstall)
# githubinstall("AutoAnnotation")
source("0_settings.R")
library(AutoAnnotation)

#-------- Set up file folders, different for different projects ----------
filenames <- GetFilenames()
dataFileName <- filenames$Outputfiles$AnimalStudiesFile
didctionaryName <- "data/ROBRegularExpression.txt"

#-------- read in data ----
animalStudies <- read.csv(dataFileName, stringsAsFactors = F)

annotationResults <- CountTermsInStudies(searchingData = dataFileName
                                         , dictionary = didctionaryName
                                         , linkSearchHeaders = "textfilename"
                                         , ignoreCase = T)

annotationOnlyResults <- as.data.frame(lapply(annotationResults[, -1],function(x) as.numeric(as.character(x))))
print(colSums(annotationOnlyResults))

# -------- write output data -----------
outputData <- cbind(animalStudies, annotationOnlyResults > 0)

write.csv(outputData, filenames$Outputfiles$AnnotatedFile, row.names = F)
