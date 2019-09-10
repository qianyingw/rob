#---------- Prepare data for MND study -------------------------------------------------
# Script to run animal identification for pmc data
# Created by Jing Liao
# Created on May 01 2019

# ---------- Load functions and libraries --------------------------------------
source('functions/syrfDBReader.R')
source('functions/ioeAPI.R')
source('functions/syrfIoeHelper.R')
source('functions/analysis.R')
source("functions/configure.R")
source('0_settings.R')

library(AzureStor)
library(magrittr)
library(rlang)
library(httr)
library(RCurl)
library(rjson)
library(tidyr)

# ---------- Inputs ------------------------------
# runTimestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
projectName <- "Identification of animal experiments"
filenames <- GetFilenames()

#----------- Set up output file name -------------------------
#Preapre the training data and testing data
fileFolders <- CreateFileFolders(filenames$Inputfiles$DataFolder, filenames$Outputfiles$AnimalStudiesFolder)

# --- 1. Read data from database ----
# extract project matching the given project names
syrfConnection <- SyRFConnection()
projects <- GetProjects(syrfConnection$conProject)
myProject <- projects[which(projects$Name == projectName),]
myProject <- GetProject(syrfConnection$conProject, myProject$`_id`[[1]], raw = T)
print(myProject$Name)
# read studies of the project
myStudies <- GetStudiesForProject(syrfConnection$conStudy, myProject$`_id`[[1]], raw = TRUE)
myStudies$Cat <- ""
myStudies$Label <- CalculateDecisionForIOE(myStudies, myProject)
print(table(myStudies$Label))

#---- 2. Process data for IOE API ----
# 0: excluded, "": unknown, 1: included
# Take out data from xml file and convert it for ioeAPI
nPatch <- 20
fullList <- read.csv(filenames$Outputfiles$BulkXMLTiabFile, row.names = F)
ioeApiDataPMC <- fullList[fullList$textfilename != "",]
ioeApiDataPMC <- ioeApiDataPMC %>%
  mutate(
    LABEL = 99,
    ITEM_ID = pmcid,
    REVIEW_ID = myProject$idStr, 
    TITLE = Title,
    KEYWORDS = "",
    ABSTRACT = Abstract,
    Cat = ""
  ) %>%
  select(
    "LABEL"  ,   "ITEM_ID"  , "REVIEW_ID" ,"TITLE"   ,  "KEYWORDS" , "ABSTRACT", "Cat" 
  )
set.seed(42)
ioeApiDataPMC$patch <- sample(nPatch, nrow(fullList), replace=T)
table(ioeApiDataPMC$patch)
# Extract data from studies 
ioeApiData <- ExtractDataForIoeAPI(myStudies)
print(table(ioeApiData$LABEL))

#---- 2. Divide data into 20 patches, combine screened data with each patch, send to API, and analyze the results  ----
results <- data.frame(score = double()  , ITEM_ID = character() ,  REVIEW_ID = character())
for(i in 1:nPatch){
  idata <- ioeApiDataPMC[which(ioeApiDataPMC$patch == i),1:7]
  combineData <- rbind(ioeApiData, idata)
  outputFilenames <- CreateMLFilenames(fileFolders$OutputFolder, i)
  
  # Write data and label files out as IOE API protocol
  allDecisions <- WriteFilesForIOE(combineData, outputFilenames)
  
  # Send the data to IOE API and Write the results
  start_time <- Sys.time()
  
  ifilenames <- CreateFileNamesForIOEAPI(outputFilenames$Records, outputFilenames$Decisions
                                        , outputFilenames$Vectors ,outputFilenames$Results)
  TrainCollection(ifilenames, gsub("[-]","", paste0(myProject$idStr, i)))

  iresults <- read.csv(ifilenames$ResultsFileName, header = FALSE, col.names = c("score", "ITEM_ID","REVIEW_ID"))
  results <- rbind(results, iresults)
  end_time <- Sys.time()
  runningTime <- end_time - start_time
  print(paste0("running time: ", runningTime))
  
  # Analyse training results
  analysisResult <- Analysis(outputFilenames$Results, outputFilenames$TestDecisions, outputFilenames$Analysis)
}

threshold <- 0.15
index <- which(grepl("PMC", results$ITEM_ID) & results$score > threshold)
animalStudies <- results[index,] %>%
    left_join(xmlFileLists, by=c("ITEM_ID" = "pmcid"))

write.csv(animalStudies, file = filenames$Outputfiles$AnimalStudiesFile, row.names = F)
