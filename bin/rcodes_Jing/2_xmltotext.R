source('0_settings.R')
library(xml2)
library(stringi)

filenames <- GetFilenames()
if(file.exists(filenames$Outputfiles$BulkXMLFile ))  xmlFileLists <- read.csv(filenames$Outputfiles$BulkXMLFile, row.names = F)

xmlFileLists$filePathFull <- paste0(filenames$Outputfiles$BulkFolder,  xmlFileLists$filePath)

GetTextWithName <-function(xmlFileContent, name){
  text <- ""
  
  try(
    text <- xml_text(xml_find_all(xmlFileContent,name))[1]
  ) 
  return(text)
}

GetTitle <- function(filePathFull, name = ".//article-title"){
  xmlFileContent <- read_xml(filePathFull)
  text <- GetTextWithName(xmlFileContent, name)
  return(text)
}

GetAbstract <- function(filePathFull, name = ".//abstract"){
  xmlFileContent <- read_xml(filePathFull)
  text <- GetTextWithName(xmlFileContent, name)
  return(text)
}

GetFullText <- function(filePathFull, name = ".//body"){
  xmlFileContent <- read_xml(filePathFull)
  text <- GetTextWithName(xmlFileContent, name)
  return(text)
}

ExtractFullTextToFile <- function(filePathFull)
{
  textfilename <- gsub(".nxml",".txt", filePathFull)
  if(file.exists(textfilename))  return(textfilename)
  else textfilename <- ""
   try({
  stri_write_lines(GetFullText(filePathFull), textfilename) 
  return(textfilename )}
  )
  return("")
}

xmlFileLists$Title <- sapply(xmlFileLists$filePathFull, GetTitle)
xmlFileLists$Abstract <- sapply(xmlFileLists$filePathFull, GetAbstract)

# xmlFileLists$FullText <- sapply(xmlFileLists$filePathFull, GetFullText)
xmlFileLists$textfilename <- sapply(xmlFileLists$filePathFull, ExtractFullTextToFile)

write.csv(xmlFileLists, file = filenames$Outputfiles$BulkXMLTiabFile)

# xmlFileLists$Title <- sapply(xmlFileLists$Title,  function(x) return(x[1])) 
# xmlFileLists$Abstract <- sapply(xmlFileLists$Abstract,  function(x) return(x[1])) 
