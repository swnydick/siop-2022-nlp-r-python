#########################################
# NLP in R via Python                   #
#                                       #
# Korn Ferry Institute: Automation Team #
# 2022-04-15                            #
#########################################


# 1. Setup / Required Packages =================================================
require(caret)
require(ggplot2)
require(keras)
require(roperators)
require(sentiment.ai)
require(sentimentr)
require(SnowballC) # for tm
require(tfhub)
require(tm)


# run once and restart R session!
# this will set up all the dependencies you need to run keras and sentiment.ai
# install_sentiment.ai()

# initialise the environment
init_sentiment.ai()


# source tf hub py function
reticulate::source_python("exercises/get_embedder.py")

# load python function
embedder <- load_language_model("https://tfhub.dev/google/universal-sentence-encoder/4")
# OR use tfhub package (both do the same thing and return python functions)
# difference is tfhub doesn't let you specify where to cache the model! 
# embedder <- tfhub::hub_load("https://tfhub.dev/google/universal-sentence-encoder/4")
# For large datasets, sentiment.ai::embed_text() has batch control

# prep the data we will use

# turn response sentiment into numeric data
tweet_sentiments <- sentiment.ai::airline_tweets$airline_sentiment |>
                    factor(levels = c("negative", "neutral", "positive"), 
                           labels = -1:1) |>
                    f.as.numeric()

# 2. Explore Embeddings ========================================================

# test!
words <- c("Dog", "David Bowie", "Magic Dance", "Good", "Lady Gaga", "Labrador")

# use python function
embeddings <- embedder(words)

# need to convert to R object
embeddings <- as.matrix(embeddings)
rownames(embeddings) <- words

embeddings[, 1:6]

# find similaritry
sentiment.ai::cosine(embeddings)

# To get ranked matches
matched <- sentiment.ai::cosine_match(embeddings, embeddings)

# to find the next most similar (since self-self will have similarity of 1!)
best_matches <- matched[rank == 2] 
best_matches

# Can plot (invert similarity! & remove symetric duplicates)
best_matches[, isimilarity := 1-similarity]
psort <- function(x,y) paste(sort(c(chr(x),chr(y))), collapse = " ")
best_matches[, reps := psort(target, reference), by = "target"]
best_matches <- best_matches[!duplicated(reps)]
data.table::setorder(best_matches, similarity)
best_matches[, target := factor(target, levels = target)]

best_matches |>
  ggplot(aes(x = target, y = isimilarity)) + 
  geom_segment(aes(xend = target, 
                   y = 0, 
                   yend = isimilarity),
               arrow = arrow(length = unit(0.5, "cm"), 
                             type = "closed"),
               color = "#00634F",
               size = 1.2) +
  geom_text(aes(label = reference), 
            size = 5, 
            hjust = -0.2, 
            color = "#568E30") +
  coord_flip() +
  theme_void() + 
  expand_limits(y=c(0, 1.1)) +
  theme(axis.text.y = element_text(size = 12, 
                                   face = "bold", 
                                   hjust = 1.2, 
                                   color = "#06332A"),
        plot.margin = margin(1,1.5,1,1.5, "cm"))


# can even plot text similarity with a PCA!
embeddings |> 
  t() |>
  princomp() |>
  biplot(cex = c(.2, 1.5), 
         xlim=c(-.2, .2), 
         ylim=c(-.2, .2),
         col = c("#000000", "#920A7A"))
# 3rd dimension probably matters here!

# Now to embed airline tweets

# embed text. 
# airline_tweets is a dataset in sentiment.ai
# need to turn into matrix as result is python data type! 
tweet_embeddings <- embedder(airline_tweets$text) |> as.matrix()



# 3. Creating Corpus/IDF =======================================================

# Now, create matrix of term counts to compare against embeddings
# Function and explanation is in make_corpus.R
# code used last year!
source("exercises/make_corpus.R")

# create a corpus per row of text in the data. a corpus is a format for storing
# text data.. often contains meta information (but doesn't have too)

# VectorSource - tells R to treat each element as if it were a document
# SimpleCorpus - function that turns the text in to corpora
text_corpus <- make_corpus(airline_tweets$text)


# Bag of words approach basically ignores sentence grammer, word order, etc. 
# How we have set up our data thus far is inline w/ the bag of words approach.
# We have created Term Frequency matrix - often times for modeling, we create 
# something called the Term Frequency - Inverse Document Frequency Matrix. This
# matrix weights Term Frequency by how prevalent they are in the corpora
# (text strings). If the word shows up often in and across the documents - they 
# get less weight (for example stop words - however we have already removed
# these). A good overview can be found at: http://www.tfidf.com/

# Note: Sparse terms are removed - here we want as close to 512 featyures as possible
# of comments
tweet_tfidf <- DocumentTermMatrix(x       = text_corpus, 
                                  control = list(weighting = weightTfIdf)) |>
                removeSparseTerms(sparse = .99699) |>
                as.matrix()

# Now we have an IDF matrix of words per tweet
tweet_tfidf[1:10, 1:10]

# 4. Create Models==============================================================


# Here we will test 3 different approaches to model tweet sentiment

# 1. Neural network (via keras) on embedding matrix
# 2. Neural network (via keras) on bag of words/IDF matrix
# 3. Lexical sentiment analysis. 

# Step 1: set seed and create test/train split.
# this ensures reproducability and a fair comparison 
set.seed(608512) # for R
reticulate::py_set_seed(345877) #for Python
tensorflow::set_random_seed(246658, disable_gpu = TRUE) # for tensorflow (via python)


# now make a 70:30 test/train split with caret
# use logical condition to make test/train more fair
train_idx <- caret::createDataPartition(y    = tweet_sentiments == 1,
                                        p    = 0.7,
                                        list = FALSE)


# create data partitions
y_train   <- tweet_sentiments[train_idx]  |> as.matrix()
y_test    <- tweet_sentiments[-train_idx] |> as.matrix()

emb_train <- tweet_embeddings[train_idx, ]
emb_test  <- tweet_embeddings[-train_idx, ]

idf_train <- tweet_tfidf[train_idx, ]
idf_test  <- tweet_tfidf[-train_idx, ]


# 4a Embedding model -----------------------------------------------------------
# USE embeddings are 512 dimensional
emb_input  <- layer_input(shape = 512, name  = "input")

emb_hidden <- emb_input |>
              layer_dense(units = 64, 
                          activation = "tanh", 
                          name       = "hidden0",  
                          kernel_regularizer = regularizer_l1_l2(l1 = 0.005,
                                                                 l2 = 0.005)) |>
              layer_dense(units = 32, 
                          activation = "tanh", 
                          name       = "hidden1")

# Output - sigmoid for probabilities
emb_out   <- layer_dense(object     = emb_hidden,
                         units      = 1, 
                         activation = "tanh",
                         name       = "output")

emb_model <- keras_model(emb_input, emb_out)

compile(emb_model,
        optimizer = optimizer_adam(),
        loss      = "mean_squared_error",
        metrics   = c("mean_absolute_error", "cosine_proximity")
        )


# sometimes the callbacks argument will not work
emb_history <- fit(object          = emb_model,
                   x               = emb_train,
                   y               = y_train,
                   batch_size      = 1024,
                   validation_data = list(emb_test, y_test),
                   epochs          = 100,
                   shuffle         = TRUE,
                   view_metrics    = TRUE,
                   verbose         = 1)

# look at training results
print(emb_history)
emb_predictions <- predict(emb_model, emb_test)
emb_cor         <- cor(emb_predictions, y_test)

# 4b. TFIDF model --------------------------------------------------------------

idf_input  <- layer_input(shape = ncol(tweet_tfidf), name  = "input")

idf_hidden <- idf_input |>
  layer_dense(units = 64, 
              activation = "tanh", 
              name       = "hidden0",  
              kernel_regularizer = regularizer_l1_l2(l1 = 0.005,
                                                     l2 = 0.005)) |>
  layer_dense(units = 32, 
              activation = "tanh", 
              name       = "hidden1")

# Output - sigmoid for probabilities
idf_out   <- layer_dense(object     = idf_hidden,
                         units      = 1, 
                         activation = "tanh",
                         name       = "output")

idf_model <- keras_model(idf_input, idf_out)

compile(idf_model,
        optimizer = optimizer_adam(),
        loss      = "mean_squared_error",
        metrics   = c("mean_absolute_error", "cosine_proximity")
)


# sometimes the callbacks argument will not work
idf_history <- fit(object          = idf_model,
                   x               = idf_train,
                   y               = y_train,
                   batch_size      = 1024,
                   validation_data = list(idf_test, y_test),
                   epochs          = 100,
                   shuffle         = TRUE,
                   view_metrics    = TRUE,
                   verbose         = 1)

# look at training results
print(idf_history)
idf_predictions <- predict(idf_model, idf_test)
idf_cor         <- cor(idf_predictions, y_test)

# 4c Lexical Sentiment analysis ------------------------------------------------

test_tweets <- airline_tweets$text[-train_idx]

lex_predictions <- sentiment_by(get_sentences(test_tweets), 
                                1:length(test_tweets))$ave_sentiment

lex_cor         <- cor(lex_predictions, y_test)

# 4d. Evaluate ----------------------------------------------------------------- 


eval_df <- data.frame(model = c("Embedding","IDF matrix", "Lexical"),
                      r     = c(emb_cor, idf_cor, lex_cor))

ggplot(eval_df, aes(y = model, x = r)) + 
  geom_col(fill = "#00634F", width = .5) + 
  theme_minimal()

# NOTE: The IDF-based model will be difficult to generalize to other data!
#       The embedding model will also perform slightly worse if applied to 
#       another context, however should still be useable! Sentimentr is at a 
#       disadvantage since the other models were trained for airline tweets! 
