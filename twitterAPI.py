'''
this is bob the code support dog
bob is here to help you cope with the code below

                 _.--"""--._
                      .'       '-. `.
                   __/__    (-.   `\ \
                  /o `o \      \    \ \
                 _\__.__/ ))    |    | ;
            .--;"               |    |  \
           (    `)              |    |   \
          _|`---' .'      _,   _|    |    `\
        '`_\  \     '_,.-';_.-`\|     \     \_
        .'  '--'---;`  / /     |\     |_..--' \
                   \'-'.'     .--'.__/    __.-;
                    `"`      (___...---''`     \
                             _/_                \
                            /bob\
                            \___/


'''
# Fetch tweets from given query and save them as comma separated file
# organized by id, date of creation and tweet content

import tweepy
import time
import csv

# 15 minutes time interval due API limits
INTERVAL = 900
N_TWETES = 3000 # number of tweets to search and download

# keys and tokens
consumer_key = FILL
consumer_secret = FILL
access_secret = FILL
access_token_secret = FILL

# set up OAuth and integrate with API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_secret, access_token_secret)
api = tweepy.API(auth)


# check for dupes
def remove_dupes(all_twetes):
    twete_checklist = []

    for twete in all_twetes:
        if twete not in twete_checklist:
            twete_checklist.append(twete)

    return twete_checklist

# intentional twetes misspelling
def fetch_twetes(query=None):
    all_twetes = []
    try:
        twetes = api.search(q=str(query), lang="en", result_type="recent")

        all_twetes.extend(twetes)

        last_twete = all_twetes[-1].id - 1
    except tweepy.TweepError:
        print("Waiting for the 15 minutes rate limit interval")
        time.sleep(INTERVAL)

    # create CSV file and header
    with open("dataset/%s-tweets.csv" % query.split(' ')[0],
              "w") as \
            twetes_file:
        writer = csv.writer(twetes_file)

        print("Fetching tweets")
        while len(all_twetes) < N_TWETES:
        # fetches tweets until tweepy rate limit reached then sleeps for
        # 15 minutes due API limits
            try:
                # fetch tweets
                twetes = api.search(q=str(query),
                                    lang="en",
                                    result_type="recent",
                                    since_id=last_twete)

                all_twetes.extend(twetes)

                # cache last tweet id
                last_twete = all_twetes[-1].id - 1

                print("%s tweets fetched " % len(all_twetes))

            # if rate limit is reached sleep for 15 minutes
            except tweepy.TweepError:
                print("Waiting for the 15 minutes rate limit interval")
                time.sleep(INTERVAL)
                continue


        unique_twetes = remove_dupes(all_twetes)

        # format tweets and IDs in columns
        formatted_twetes = [[twete.id_str,
                             twete.text.encode("utf-8")] for twete in
                            unique_twetes]

        # write N tweets to CSV
        writer.writerows(formatted_twetes)
        print("Tweets written")

    pass

if __name__ == "__main__":
    fetch_twetes("XLM")
