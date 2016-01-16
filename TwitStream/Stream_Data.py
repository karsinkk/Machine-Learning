#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
access_token = "2721199920-07tyiWqzs3AhVSL9VcysgSkd7vt5SkpmYt8rksp"
access_token_secret = "th83lkpevH0OZ6mLBwDMFLOhZxB8rn7hPvRLUNqxmH6XU"
consumer_key = "zLTU648jvGnli2enTImW1T3Oy"
consumer_secret = "68aRk65sHxO2n21rmc7D3Pv282RUgwTkCcgS5BaP84sHxXWUdf"


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    stream.filter(async=True)