
GetFilenames <- function(baseUrl = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/"
                         ,   oaFileList = "oa_file_list.csv"
                         , commFileList = "oa_comm_use_file_list.csv"
                         , nonCommFileList = "oa_non_comm_use_pdf.csv"
                         ,   dataFolder = "data/"
                         , outputFolder = "output/"
                         , bulkFolder = "oa_bulk/"
                         , oa_package = "oa_package/"
                         , animalStudiesFolder = "oa_bulk/animalFilter/"){
  inputfiles <- list(BaseUrl = baseUrl,
                     OaFileList = oaFileList,
                     DataFolder = dataFolder,
                     BulkFolder = bulkFolder,
                     BulkFolderUrl = paste0(baseUrl, bulkFolder),
                     OaListUrl = paste0(baseUrl, oaFileList),
                     NonCommListUrl = paste0(baseUrl, nonCommFileList),
                     CommListUrl = paste0(baseUrl, commFileList))
  
  outputfiles <- list(  OutputFolder = outputFolder,
                        BulkFolder = paste0(outputFolder, bulkFolder),
                        Oa_packageFolder =  paste0(outputFolder, oa_package),
                        LocalOaList = paste0(outputFolder, oa_package, oaFileList),
                        LocalNonCommList = paste0(outputFolder, oa_package, nonCommFileList),
                        LocalCommList = paste0(outputFolder, oa_package, commFileList),
                        CombinedFileList = paste0(outputFolder, oa_package, "CombinedFileList.csv"),
                        XMLFile = paste0(outputFolder, oa_package, "xmlFile.csv") ,
                        BulkXMLFile = paste0(outputFolder, bulkFolder, "xmlFile.csv"),
                        BulkXMLTiabFile = paste0(outputFolder, bulkFolder, "xmlFileTiab.csv"),
                        AnimalStudiesFolder = animalStudiesFolder,
                        AnimalStudiesFile = paste0(outputFolder, bulkFolder,"AnimalStudiesFile.csv"),
                        AnnotatedFile =  paste0(outputFolder, bulkFolder,"AnnotatedFile.csv")
  )
  
  for(file in outputfiles) dir.create(dirname(file), recursive = T, showWarnings =F)
  
  return(list(Inputfiles = inputfiles, Outputfiles = outputfiles))  
}