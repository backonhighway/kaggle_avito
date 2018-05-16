import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

#tf_vector = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
#               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
#               smooth_idf=1, sublinear_tf=1, stop_words="english" )
#tf_vector.fit(list(X_train) + list(X_valid))
#comment_train = tf_vector.transform(X_train)
#comment_val = tf_vector.transform(X_valid)