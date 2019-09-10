require(mongolite)

JTBlob <- function()
{
  structure(
    list(
      endpoint = "https://er4ml.blob.core.windows.net",
      sas = "st=2018-11-23T15%3A00%3A49Z&se=2020-12-31T15%3A00%3A00Z&sp=rwdl&sv=2018-03-28&sr=c&sig=U9mhgC8hgEs1hQhfJ3jXun6vg%2BJqqvkS3P1FxqgOEVw%3D",
      container = "syrf")
    ,
    class = "JTBlob"
  )
}

JTAPI <- function()
{
  structure(
    list(
      VectoriseBaseUrl = "https://europewest.services.azureml.net/workspaces/e7258df1126f4b41ac56f7ef67d23146/services/2dc54c9bb1fb466da12c616c9bbd370a/jobs",
      VectoriseAPIKey = "nPfCOZtNCjozV49mdy1qZlYpruzT22QlhhDrJjSF6eazT0cfNVYLVD1YU7ZvihCereuNBx5UmnQlT7dQhBKqbw==",
      ScoreBaseUrl = "https://europewest.services.azureml.net/workspaces/e7258df1126f4b41ac56f7ef67d23146/services/5d5fa9e608b245aeb481150b29e710c2/jobs",
      ScoreApiKey = "0GrP/egb80etojoJNV2n/lKC3g4/F17wgfJ4Jkmmo5YtlOaLyN4irJt1OwfHhzw2l/o+3arijj4l/odL+ib/8A=="
  )
  ,
  class = "JTAPI"
  )
}

SyRFConnection <- function(){
    dbname = 'syrftest'
    connectionString = "mongodb://ranalysis:3xjUGzX03F8iKHqj@cluster0-shard-00-00-siwfo.mongodb.net:27017,cluster0-shard-00-01-siwfo.mongodb.net:27017,cluster0-shard-00-02-siwfo.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin"
    
    structure(
      list(
         conProject = LoadCollections("pmProject", dbname, connectionString),
        conStudy = LoadCollections("pmStudy", dbname, connectionString)
         )
      ,
      class = "SyRFConnection"
    )
  }

