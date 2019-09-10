#---------- Clean up data from Meta-analysis Data Base (ACCESS)-----------
# Script to prepare data for extracting structure data for Risk of Bias Items
# Input it csv file obtain from publication tables from ACCESS
# Output is txt file with ROS Items and pdf files copied to
# Created by Jing Liao
# Created on Oct 18 2016
#-------------------------------------------------------------------------

wd = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/codes/data_process/strokeR/'
setwd(wd)


#---- Load functions ----
source('functions.R')
source('PdfTextProcesser.R')
library(stringi)
library(tm)
#---------- Enviorment set up ---------------------------------
dir.create("data", showWarnings = FALSE)
dir.create("output", showWarnings = FALSE)

#---------- Basic Setting ------------------------------
# inputPdfFolder <- "S:/TRIALDEV/CAMARADES/"
# outputPdfFolder <- "C:/Users/jliao/Publications/"
inputPdfFolder <- 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/'
outputPdfFolder <- wd

dir.create(outputPdfFolder, showWarnings = FALSE)

filename <- "ROB Query - tblPublicationData"

filenameRead <- paste("data/", filename, ".csv", sep = "")

filenameFull <- paste("output/strokeTable_FullUniqueRecord", ".txt", sep = "")
filenameUnique<- paste("output/strokeTable_UniqueRecord_ROB", ".txt", sep = "")
filenameNotValid <- paste("output/strokeTable_NotValidDocumentLinks", ".txt", sep = "")
projectList <- paste("output/strokeTable_ProjectList",".txt", sep = "")

#---------- Read Data from File ------------------
myData <-read.csv(filenameRead, header = TRUE, sep = ",", dec = ".", fill = TRUE, comment.char = "")

# strokeData$Method.of.allocation.concealment <- as.character(strokeData$Method.of.allocation.concealment)
#
# strokeData$Method.of.blinded.assessments <- as.character(strokeData$Method.of.blinded.assessments)
#
# strokeData$Method.of.Randomisation <- as.character(strokeData$Method.of.Randomisation)

#---------- Take out unwanted data --------------------
# Take out records with no DocumentLink
myData <- myData[which(myData$DocumentLink != ''), ]  # 10298

# Take out clinical trail records
myData <-myData[-grep("MS Clinical trial", myData$DocumentLink, ignore.case = T),]  # 9125

myData <-myData[-grep("MND Clinical trial", myData$DocumentLink, ignore.case = T),]  # 8866

colnames(myData)
#------------------------------ 18 columns of the  myData --------------------------------------------------------------------
# [1] "PublicationID"                                "DocumentLink"                                 "Type.of.Disease"
# [4] "User"                                         "Pub.ID"                                       "Random.Allocation.to.Group"
# [7] "Blinded.Induction.of.Ischaemia"               "Blinded.Assessment.of.Outcome"                "Blinded.Assessment.of.Infarct.Volume"
# [10] "Blinded.Assessment.of.Neurological.Score"     "Sample.Size.Calculation"                      "Explantion.of.Animal.Exclusions"
# [13] "Use.of.Comorbid.Animals"                      "Compliance.with.Animal.Welfare.Regulations"   "Statement.of.Potential.Conflicts.of.Interest"
# [16] "Method.of.Randomisation"                      "Method.of.allocation.concealment"             "Method.of.blinded.assessments"
#-----------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------- investigate the duplication of records, which may be due to ----------------------------
# 1. double screen
# 2. multiple drugs, models, strains. Way to tell the differences: user, but most of the records don'#t have user info
#------------------------------------------------------------------------------------------------------------------------------------
myData$flag <- with(myData, as.numeric(factor(DocumentLink, levels = unique(DocumentLink))))

temp <- table(myData$flag)

myData$flagCount  <- sapply(myData$flag, function(x) temp[names(temp) == x][1])

rm(temp)
#----------------------------------------------------------------------------------
# number of duplicate     1    2    3     4    5    6    7     8    9   10   11   14   16 
# number of publication  2928 4866  438  292   90   90   49   24   18   30   11   14   16 
#----------------------------------------------------------------------------------
uniqueData <-  myData %>%
  group_by(DocumentLink, Pub.ID) %>%
  summarise(
  PublicationID = paste0(PublicationID, collapse=", ")
  ,
  RandomizationTreatmentControl = sum(abs(Random.Allocation.to.Group))  > 0
  ,
  AllocationConcealment = sum(abs(Blinded.Induction.of.Ischaemia))     > 0
  ,
  BlindedOutcomeAssessment = sum(abs(
    Blinded.Assessment.of.Outcome + Blinded.Assessment.of.Infarct.Volume + Blinded.Assessment.of.Neurological.Score
  )) > 0
  ,
  SampleSizeCalculation = sum(abs(Sample.Size.Calculation)) > 0
  ,
  AnimalExclusions = sum(abs(Explantion.of.Animal.Exclusions)) > 0
  ,
  Comorbidity = sum(abs(Use.of.Comorbid.Animals)) > 0
  ,
  AnimalWelfareRegulations = sum(abs(Compliance.with.Animal.Welfare.Regulations)) > 0
  ,
  ConflictsOfInterest = sum(abs(Statement.of.Potential.Conflicts.of.Interest)) > 0 
  ,
  TypeOfDisease = paste0(Type.of.Disease, collapse=", ")
)

#------------------ Clean up document links-----------------------
uniqueData$DocumentLink <- gsub("//", "/", uniqueData$DocumentLink, perl = T)

uniqueData$DocumentLink <- gsub("#", "", uniqueData$DocumentLink)

uniqueData$DocumentLink <- gsub("transgenic/","transgenic/All TG PDFs/TG article new/",uniqueData$DocumentLink, perl = T )

uniqueData$DocumentLink <- gsub("transgenic/All TG PDFs/TG article new/Transgenic articles/", "transgenic/All TG PDFs/Transgenic articles/",uniqueData$DocumentLink, perl = T)

uniqueData$fileLink <- paste(inputPdfFolder, uniqueData$DocumentLink, sep = "")

uniqueData$fileLink <- gsub("U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/http", "http",uniqueData$fileLink, perl = T)

#------------------ Extract information of folders of the pdfs, which provide information of which project it belongs to -------
uniqueData$Project <- sapply(as.character(uniqueData$DocumentLink), LocateFolder)

#----------------- locate all txt files in text folder ----------------------------------------
uniqueData$fileExist <- file.exists(uniqueData$fileLink)
# table(uniqueData$fileExist)
# FALSE  TRUE
# 169 (119 are http links)  5480
# 173 5476 (qwang, 10th Sep 2019)

#----------------- prepare output tab-separated text file for structure data extraction ----------------------------------------
outputStructureData <- F
if(outputStructureData)
{
#---- copy files
  uniqueData <- uniqueData %>%
    mutate(
      fleLinkNew = ifelse(fileExist
                           , paste0(outputPdfFolder, DocumentLink)
                           , NA),
      folderCreated = ifelse(fileExist
                             , DirectoryCreation(dirname(fleLinkNew))
                             , NA),
      copyFlag = ifelse(fileExist
                        , file.copy(fileLink, fileLinkNew, overwrite = F)
                        , NA)
      
    )
#---- check http links and copy valid url to fileLinkNew
indexHttp <- grep("http", uniqueData$fileLink)
# 121 links in total as file Links
uniqueData$fileExist[indexHttp] <- sapply(uniqueData$fileLink[indexHttp], validURL)
# table(uniqueData$fileExist[indexHttp])
# FALSE  TRUE 
# 12     107 

# table(uniqueData$fileExist)
# FALSE  TRUE 
# 62  5587 

#---- summary of the data
  # The index of data that contains at least one of the risk of bias terms and has valid file (pdf or url)
  indexROBPositive <- ( (uniqueData$fileExist) * (rowSums(uniqueData[, 4:11])>0  )) > 0
  table(indexROBPositive)
  # FALSE  TRUE 
  # 2183   3466 
  # summary of data ROB
  print(colSums(uniqueData[indexROBPositive,4:11]))
  # RandomizationTreatmentControl         AllocationConcealment      BlindedOutcomeAssessment 
  # 1191                           292                          1297 
  # SampleSizeCalculation              AnimalExclusions                   Comorbidity 
  # 89                           168                           136 
  # AnimalWelfareRegulations           ConflictsOfInterest 
  # 2930                                              605
  
#---- write data
  write.table(uniqueData[,c(1:11,13)], file = filenameFull, append = FALSE, sep = "\t", row.names = F)
  
  write.table(uniqueData[indexROBPositive,c(1:11,13)], file = filenameUnique, append = FALSE, sep = "\t", row.names = F)
  
  write.table(uniqueData[which(!uniqueData$fileExist), 1:3],file = filenameNotValid,append = FALSE, sep = "\t", row.names = F)
  
  write.table(as.vector(unique(uniqueData$Project)),file = projectList,append = FALSE, sep = "\t", row.names = F)
}

#-------------------- Write data for stroke ------------------------------
outputStrokeData <- F
if(outputStrokeData)
{
  strokeList  <- "Focal|Global|ICH|SAH|(Subarachnoid[ ]Haemor)|Lacunar|antidepressant"
  index4 <- ((uniqueData$fileExist) * grepl(strokeList, uniqueData$TypeOfDisease))>0
  write.table(uniqueData[index4,c(1,12,14)], file = "output/focal_unique.txt", append = FALSE, sep = "\t", row.names = F)
}

outputTransgenicData <- F
if(outputTransgenicData)
{
  index5 <- (uniqueData$Project == "transgenic")
  write.table(uniqueData[index5,], file = "output/transgenic_unique.txt", append = FALSE, sep = "\t", row.names = F)
}

#-------------------- Write data for machine learning ------------------------------
excludingDocumentLinks <- c(
  " Publications/Antidepressants_Focal/354_Jolkkonen_2000.pdf"
  , "ALS Interlib loans/26641 Alvarez.pdf"
  , "Publications/MND - preclinical/Zhou2013.pdf"
  , "Publications/MND - preclinical/2757990227/Huang-2006-[Effect of transplantation of wild-.pdf"
  , "Publications/Cancer+Pain/Bone_cancer_pain_080811/18_Han_2010.pdf"
  , "Publications/SCI+Decompression/Relevant but No Data/82_Chen_2000.pdf"
  , "Glioma/Gene Therapy/Glioma PDFs/260_Liau_1998.PDF"
  , "Publications/ICH/576_2006_J Neurosurg_Hua Y.PDF"
  , "Publications/SAH/0940711560/2006_[Evaluation of cerebral vasospasm resulting from subarachnoid hemorrhage with 1H-magnetic resonance spectroscopy].pdf"
  , "Publications/SAH/3029974030/2000_Azathioprine and methylprednisolone prevention of chronic cerebral vasospasm in dogs.pdf"
  , "Publications/SCI+Hypothermia/1797_Wang,D_2010 (mandarin).pdf"
  , "Glioma/Gene Therapy/Glioma PDFs/898 Li 2007.pdf")

validData <- uniqueData %>%
  filter(fileExist & !grepl("http", fileLink) & !(DocumentLink %in% excludingDocumentLinks)) %>%
  group_by(DocumentLink) %>%
  summarise(
    PublicationID = paste0(PublicationID, collapse=", "),
    RandomizationTreatmentControl = as.numeric(sum(RandomizationTreatmentControl) > 0),
    AllocationConcealment = as.numeric(sum(AllocationConcealment) > 0),
    BlindedOutcomeAssessment = as.numeric(sum(BlindedOutcomeAssessment) > 0),
    SampleSizeCalculation = as.numeric(sum(SampleSizeCalculation) > 0),
    AnimalExclusions = as.numeric(sum(AnimalExclusions) > 0),
    Comorbidity = as.numeric(sum(Comorbidity) > 0),
    AnimalWelfareRegulations = as.numeric(sum(AnimalWelfareRegulations) > 0),
    ConflictsOfInterest = as.numeric(sum(ConflictsOfInterest) > 0),
    fileLink = first(fileLink)) 

validData$TextFilePath<- sapply(validData$fileLink, ConvertToText)

validData <- validData %>%
  filter(!is.na(TextFilePath)) %>%
  mutate(FullTextSize = GetFileSize(TextFilePath)) 

LongValidData <- validData %>%
  filter(FullTextSize > 500)  %>%
  mutate(TextFullTextExt = file_ext(DocumentLink)) %>% 
  mutate(
    FullText = ReadFullText(TextFilePath),
    FullTextLength = nchar(FullText),
    CleanFullText = stri_trans_general(
      RemoveQuotationMarks(stripWhitespace(RemoveNewLine(FullText))),id="Latin-ASCII"),
    CleanFullTextSize = nchar(CleanFullText)
  )
 
LongValidData <- LongValidData%>%
  filter(CleanFullTextSize > 9000)

LongValidData$ID <- seq_along(LongValidData$DocumentLink)

library(stringi )

write.table(LongValidData[, c("ID", "CleanFullText","RandomizationTreatmentControl", "AllocationConcealment"
                              ,   "BlindedOutcomeAssessment",   "SampleSizeCalculation"
                              ,  "AnimalExclusions","Comorbidity",   "AnimalWelfareRegulations"  
                              ,"ConflictsOfInterest")], quote = FALSE, file = "output/dataWithFullText.txt"
            , append = FALSE, sep = "\t", row.names = F)

write.table(LongValidData[, c("ID", 'DocumentLink', 'fileLink',
                              "CleanFullText","RandomizationTreatmentControl", "AllocationConcealment"
                              ,   "BlindedOutcomeAssessment",   "SampleSizeCalculation"
                              ,  "AnimalExclusions","Comorbidity",   "AnimalWelfareRegulations"  
                              ,"ConflictsOfInterest")], quote = FALSE, file = "output/dataStrokeWithFullText_utf8.txt"
            , append = FALSE, sep = "\t", row.names = F, fileEncoding = "UTF-8")


write.table(LongValidData[, c("ID", "CleanFullText","RandomizationTreatmentControl", "AllocationConcealment"
                              ,   "BlindedOutcomeAssessment",   "SampleSizeCalculation"
                          ,  "AnimalExclusions","Comorbidity",   "AnimalWelfareRegulations"  
                          ,"ConflictsOfInterest")], quote = FALSE, file = "output/dataWithFullText_latin1.txt"
            , append = FALSE, sep = "\t", row.names = F, fileEncoding = "Latin1")


