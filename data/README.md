




# Twitter-Drug 




## Data Collection 
We first employ the official [Twitter API](https://developer.x.com/en/docs/x-api) to crawl the metadata (275,884,694 posts and 40,780,721 users) during Dec. 2020 and Aug. 2021. 
Afterward, we follow [CM-GCL](https://github.com/graphprojects/CM-GCL) to collect a keyword list that covers 56 drug types that may cause drug overdoes or drug addiction problems including opioid drugs (e.g., oxycodone, codeine, morphine, fentanyl, and heroin), hallucinogens drugs (e.g., LSD, DMT, MDMA, ketamine, and magic mushrooms), stimulant drugs (e.g., cocaine, crystal meth, and amphetamine), depressants drugs (e.g., Xanax, alprazolam, and halcion), and cannabis drugs (e.g., weed and hash oil). 
Then we filter all posts and users that contain drug-relevant information via the keyword list and further obtain 54,680 users that have 266,975 drug-relevant posts.


## Data Annotation 
After obtaining the filtered metadata from Twitter, we (including 3 groups, 2 annotators per group) have spent 62 days manually labeling the collected data based on the following rules: 
(i) If a Twitter account advertises his/her drug products during the time period by posting drug-relevant tweets or having rich connections (i.e, reply, mention, quote, follow, retweet, like) with other drug sellers/users, we consider him/her a drug seller. 
For example, Figure 1 shows a drug seller ''Ro***67'' on Twitter who advertises the drug products (e.g., oxycodone, Xanax, and LSD) by posting the drug hashtag and leaving his contact info (e.g., email address and Wickr id). 
(ii) If a Twitter account talks about having drugs or facing drug overdose issues, then we will view him/her as a drug user.
Figure 1 also illustrates a drug user ``cs***rt'' on Twitter who wants to try DMT after trying other drugs. 
(iii) If a Twitter account actively discusses drugs or has rich connections with other drug sellers/users but he/she has not had drugs yet, we will consider him a drug discussant. 
(iv) Some accounts may have multiple roles in the drug trafficking community, we set some specific rules to simplify the multiple roles for the sake of convenience. 
For instance, a drug seller who advertises his/her drug products frequently is also a drug user. In this case, we will identify this user as a drug seller. 
With these criteria, three groups labeled the filtered meta-data separately. 
For those labeled data that we had disagreements on, we had discussions among groups for further cross-validation. 
To conclude, we obtained $445$ drug sellers, $1,932$ drug users, $522$ drug discussants, and $25,046$ normal users. 
<div align="center">
<img src="https://github.com/graphprojects/AD-GSMOTE/blob/main/figs/data.png" width="1200", height="700" alt="Illustration of the Twitter-Drug graph dataset">
<p>Figure 1: Illustration of the Twitter-Drug graph dataset.</p>
</div>

## Twitter-Drug Graph Construction 
After obtaining the ground truth, we build a multi-relation graph called Twitter-Drug to model the rich connections among these users. 
Specifically, each user is a node in Twitter-Drug. 
To describe each user node, we concatenate all text information (including bio info and tweet info) during this time slot and further we feed the concatenated text content to the pre-trained language model SentenceBert to get the feature vector.  
In addition, in order to describe the relationships among users, we divide different relations into three categories: U-T-U, U-F-U, and U-K-U. 
The relation U-T-U links two users when one user replies/quotes/retweets/likes another's tweet. 
These connection activities are based on Tweets and we believe they have similar activities among users. 
The relation U-F-U connects two users when one user follows another user. 
The follow/following relationship shows their social connections in daily life. 
The relation U-K-U links two users if their content information contains the same keyword, which shows they may be interested in the same topics. 
Note that, instead of using all keywords, we employ Chi-Square statistic to select a set of useful keywords. The number of each relation type is listed in Table 1. 


### Data Statistic 
Table 1: Statistics of our new multi-relation graph Twitter-Drug.
|Dataset| # of class | Feature dimension | # of drug seller | # of drug user | # of drug discussant | # of normal user | # of nodes | Relation | # of relation |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
||  |  |  |  |  | | | U-T-U | 392,190 |
|Twitter-Drug| 4 | 384 |  445 | 1,932 | 522 | 25,046 | 27,945 | U-F-U | 69,675 |
||  | |  |  |  | |  | U-K-U | 253,602 |
