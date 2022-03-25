# This is the logic used to make a cleaned text corpus we used last year

make_corpus <- function(text){
  
  # create a corpus per row of text in the data. a corpus is a format for storing
  # text data.. often contains meta information (but doesn't have too)
  
  # VectorSource - tells R to treat each element as if it were a document
  # SimpleCorpus - function that turns the text in to corpora
  text_corpus <- VectorSource(text) |> 
                 VCorpus() |>
                 tm_map(content_transformer(tolower)) |> 
                 tm_map(removeNumbers) |>
                 tm_map(removePunctuation) |>
                 tm_map(removeWords, stopwords("english")) |>
                 tm_map(stripWhitespace) |>
                 tm_map(stemDocument)
  
  return(text_corpus)
}



# we will create a custom Tokenizer function to pass to tm's functions
ngram_tokenizer <- function(x, n) {
  
  # turn a text string (corpus) into a vector of words
  text_words <- words(x)
  # create a n-gram representation of the vector of words
  text_gram  <- ngrams(text_words, n = n)
  # format the text_gram in a way that the tm package functions can process
  out        <- lapply(text_gram, paste, collapse = ' ') %>%
    unlist()
  
  return(out)
}

bigram  <- function(x) ngram_tokenizer(x, n = 2)
trigram <- function(x) ngram_tokenizer(x, n = 3)
