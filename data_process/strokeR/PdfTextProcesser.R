#---------- Basic functions for text processing -----------
# Previouse call Basic functions
# Created by Jing Liao
# Created on 2017
#-------------------------------------------------------------------------
library(httr)
library(rvest)
library(tools)
library(dplyr)
# library(tm)
library(textcat)
library(textreadr)

GetTextSize <- function(text)
{
  nchar(text)
}

GetFileSize <- function(txtfilename)
{
  return(file.info(txtfilename)$size)
}

# Read full text
ReadFullText <- function(txtfilename){
  readinFulltext <- function(inputarray){  
    if(is.na(inputarray[2]) || inputarray[2] < 10)
    { 
      return ("")
    }
    else 
    {
      return (readChar(inputarray[1], inputarray[2])) 
    }
  }
  
  #--------- Calculate file size  -------------------------------------------------------------------------------
  fulltextsize <- GetFileSize(txtfilename)
  #--------- Read text from file into myData$fulltext -----------------------------------------------------------
  fulltext <- apply(cbind(txtfilename = txtfilename, fulltextsize = fulltextsize), 1, readinFulltext)
  
  return(fulltext)
}

ValidURL <- function(fileLink) {
  result <- tryCatch({

    
    foo <- !http_error(fileLink)
    return(foo)
  }
  , warning = function(war) {
    return(F)
  }
  , error = function(err) {
    return(F)
  }
  , finally = {
  })
}

DirectoryCreation <- function(path) {
  dir.create(path, showWarnings = FALSE, recursive = TRUE)
}

CopyPdf <- function(fileLink, fileLinkNew, overwirte = F){
  file.copy(fileLink, fileLinkNew, overwrite = F)
  
  return(file.exists(fileLinkNew))
}

OpenPdf <- function(pdflink){
  exe <- 'C:\\Program Files (x86)\\Adobe\\Reader 11.0\\Reader\\AcroRd32.exe'
    if(file.exists(pdflink)){  
      system(paste("\"", exe, "\" \"", pdflink, "\"", sep = ""), wait = F)
    }else
  {
    print(paste('file not found: ', pdflink))
    }
}

ConvertToText <- function(originalFileLink, ignoreExistingTextFile = FALSE){

  if(tolower(file_ext(originalFileLink)) %in% c("pdf")) return(ConvertPdfToText(originalFileLink, ignoreExistingTextFile = ignoreExistingTextFile))
  
  if(tolower(file_ext(originalFileLink)) %in% c("doc")) return(ConvertDocToText(originalFileLink, ignoreExistingTextFile = ignoreExistingTextFile))
  
  if(tolower(file_ext(originalFileLink)) %in% c("mht", "htm", "html", "asp")) return(ConvertHTMLToText(originalFileLink, ignoreExistingTextFile = ignoreExistingTextFile))
  
  if(tolower(file_ext(originalFileLink)) %in% c("xls")) return(NA)
  
  if(tolower(file_ext(originalFileLink)) %in% c("ppt")) return(NA)
  
  return(NA)
}

ConvertHTMLToText <- function(fileLink, ignoreExistingTextFile = FALSE){
  ext <- file_ext(fileLink)

  txtLink <- sub(paste0(".",ext), '.txt', fileLink, ignore.case = TRUE)
  if(file.exists(txtLink) & ignoreExistingTextFile == FALSE) 
  {
    return(txtLink)
  }
  text <- read_html(fileLink)
  if(!exists('text')) return(NA)
  
  writeLines(text, txtLink) 
  
  return(txtLink)
}

ConvertDocToText <- function(fileLink, ignoreExistingTextFile = FALSE){
  ext <- file_ext(fileLink)
  txtLink <- sub(paste0(".",ext), '.txt', fileLink, ignore.case = TRUE)
  if(file.exists(txtLink) & ignoreExistingTextFile == FALSE) 
  {
    return(txtLink)
  }
  
  text <- read_document(fileLink)
  if(!exists('text') | is.null(text)) return(NA)
  writeLines(text,txtLink) 
  return(txtLink)
}

#Convert Pdf to Text
ConvertPdfToText <- function(fileLink, ignoreExistingTextFile = FALSE){
  ext <- file_ext(fileLink)
  
  txtLink <- sub(paste0(".",ext), '.txt', fileLink, ignore.case = TRUE)
  
  if(file.exists(txtLink) & ignoreExistingTextFile == FALSE) 
  {
    return(txtLink)
  }
  exe <- '"pdftotext"'
  # exe <- '"C:\\xpdfbin-win-3.04\\bin64\\pdftotext.exe"'
  if(file.exists(fileLink))
  {
    com <- paste(exe, paste('"',fileLink,'"',sep=''))
    statusCode <- system(com, wait = T)
    
    if(statusCode == 0)  return(txtLink)
    else {
      # print(pdfLink)
      # print("----------------------------")
      return(NA)
    }
  }
  return(NA)
}

RemoveQuotationMarks <- function(text){
  text <- gsub('\"\'', " ", text)
}

RemoveNewLine <- function(text){
  text <- gsub("/(\r\n)+|\r+|\n+|\t+/i", " ", text)
}

LocateFolder <- function(DocumentLink) {
  library(stringr)
  
  if (grepl("http",
            DocumentLink,
            perl = T,
            ignore.case = T)) {
    return("")
  }
  
  if (grepl("Publications",
            DocumentLink,
            perl = T,
            ignore.case = T))
  {
    return(str_extract(DocumentLink, "(?<=Publications[/]).+?(?=[/])"))
  }
  else {
    return(str_extract(DocumentLink, "^.+?(?=[/])"))
  }
}

validURL <- function(fileLink) {
  result <- tryCatch({
    foo <- !http_error(fileLink)
    return(foo)
  }
  , warning = function(war) {
    return(F)
  }
  , error = function(err) {
    return(F)
  }
  , finally = {
  })
}

DirectoryCreation <- function(path) {
  dir.create(path, showWarnings = FALSE, recursive = TRUE)
}

DetectLangureage <- function(text){

  textcat(text)
}

OpenPdf <- function(pdfPath){
  str <- paste0('open', ' "',pdfPath,'"')
  system(str)
}