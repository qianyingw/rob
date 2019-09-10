#---------- Functions ----------------------------------------
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
