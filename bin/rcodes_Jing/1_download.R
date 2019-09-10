source('0_settings.R')

DownloadPMCfiles <-function(method = 1){
  
library(RCurl)
options(stringsAsFactors = FALSE)

filenames <- GetFilenames()

#----  Method 1 download pdfs in bulk for studies  -----
if(method == 1){
  bulkFilenames <- strsplit(getURL(filenames$Inputfiles$BulkFolderUrl, ftp.use.epsv = FALSE, dirlistonly = TRUE), "\n")[[1]]
  for(i in seq_along(bulkFilenames)){
    sourceFilename <- paste0(filenames$Inputfiles$BulkFolderUrl,  bulkFilenames[i])
    outputFilename <- paste0(filenames$Outputfiles$Oa_packageFolder,  bulkFilenames[i])
    download.file(sourceFilename, outputFilename)
    untar(outputFilename, exdir = filenames$Outputfiles$BulkFolder)
  }
  
  for(i in seq_along(bulkFilenames)){
    outputFilename <- paste0(filenames$Outputfiles$Oa_packageFolder,  bulkFilenames[i])
    untar(outputFilename, exdir =filenames$Outputfiles$BulkFolder)
    file.remove(outputFilename)
  }
  
  oaFileLists <- data.frame(filePath = list.files(path = filenames$Outputfiles$BulkFolder, recursive = T))
  oaFileLists$filePathFull <- paste0(filenames$Outputfiles$BulkFolder,  oaFileLists$filePath)
  
  xmlIndex <- grep("nxml", oaFileLists$filePath)
  
  textFileLists <- oaFileLists[setdiff(oaFileLists$Id, xmlIndex),]
  textFileLists$filePathFull <- paste0(filenames$Outputfiles$BulkFolder,  textFileLists$filePath)
  file.remove(textFileLists$filePathFull)
  
  xmlFileLists <- oaFileLists[xmlIndex,]
  xmlFileLists$pmcid <- tools::file_path_sans_ext(basename(xmlFileLists$filePath))
  xmlFileLists$Id <- seq_along(xmlFileLists$filePath)
  xmlFileLists$filePathFull <- paste0(filenames$Outputfiles$BulkFolder,  xmlFileLists$filePath)
  
  write.csv(xmlFileLists, file = filenames$Outputfiles$BulkXMLFile)
}

#----  Method 2 download packes for studies  -----
if(method == 2){
  if(!file.exists(filenames$Outputfiles$CombinedFileList)) {
    if(!file.exists(filenames$Outputfiles$LocalCommList)) download.file(filenames$Inputfiles$CommListUrl, filenames$filenames$LocalCommList)
    if(!file.exists(filenames$Outputfiles$LocalNonCommList)) download.file(filenames$Isnputfiles$NonCommListUrl, filenames$Outputfiles$LocalNonCommList)
    filelist <- rbind(read.csv(filenames$Outputfiles$LocalCommList), read.csv(filenames$Outputfiles$LocalNonCommList))
    filelist$xmlPath <- ""
    write.csv(filelist, file=filenames$Outputfiles$CombinedFileList, row.names = F, na = "")
    } else {
    filelist <- read.csv(file = filenames$Outputfiles$CombinedFileList, na.strings = c(), stringsAsFactors = F)
    filelist$xmlPath[is.na(filelist$xmlPath)]<-""
  }
   
  getXMLFile <- function(filepath, filenames, xmlPathInFile){
    if(xmlPathInFile != "" & file.exists(paste0(filenames$Outputfiles$Oa_packageFolder,  xmlPathInFile))) return(xmlPathInFile)
  
    sourceFilename <- paste0(filenames$Inputfiles$BaseUrl,  filepath)
    outputFilename <- paste0(filenames$Outputfiles$Oa_packageFolder,  filepath)
    dir.create(dirname(outputFilename), recursive = T, showWarnings = F)
    
    if(!file.exists(outputFilename)) download.file(sourceFilename, outputFilename)
    untar(outputFilename, exdir = dirname(outputFilename))
      
      fileDirectory <- gsub(".tar.gz", "", outputFilename)
      temp <- list.files(fileDirectory, recursive = T)
      index <- grep("[.]nxml", temp)
      xmlfiles <- temp[index]
      otherfiles <- temp[-index]
      file.remove(paste(fileDirectory,  otherfiles, sep="/"))
      file.remove(outputFilename)
      
    return(paste( dirname(filepath), basename(fileDirectory), xmlfiles, sep="/"))
  }

  index <- which(filelist$xmlPath == "" | file.exists(paste0(filenames$Outputfiles$Oa_packageFolder,  filelist$xmlPath)))
  for(i in seq_along(filelist$File[index])){
    print(i)
    if(filelist$xmlPath[i] == "" | !file.exists(paste0(filenames$Outputfiles$Oa_packageFolder,  filelist$xmlPath[i]))) filelist$xmlPath[i] <- getXMLFile(filelist$File[i], filenames, filelist$xmlPath[i])
    # write.csv(filelist, file=filenames$Outputfiles$CombinedFileList)
  }
  xmlFileLists <- filelist
  write.csv(xmlFileLists, file = filenames$Outputfiles$XMLFile)
}

return(xmlFileLists)
}

#----  Run dwonload file function -----
DownloadPMCfiles()