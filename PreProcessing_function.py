def PreProcessingTweets(tweet_text): #input tweet_text is list of tweets
    processed_lists=[]
    for i in tweet_text:
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',i) #substituting urls with constant URL
        tweet = re.sub('@[^\s]+','AT_USER',tweet) #substituting user mentions with constant AT_USER
        tweet = re.sub(r'#([^\s]+)',r'\1',tweet)
        processed_lists.append(tweet)
    return processed_lists
