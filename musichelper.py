from six.moves import cPickle as pickle
import numpy as np
from sets import Set 
from collections import OrderedDict
with open('members.csv', 'r') as fin:
	users = []
	user_to_index = {}
	user_song_dic = dict()
	for user_index,line in enumerate(fin):
		if user_index != 0:
			thisLine = line.split(",")
			users.append(thisLine[0])
			user_to_index[thisLine[0]] = user_index - 1
			user_song_dic[user_index-1] = Set()
	fin.close()
	print("NUM_USER" + str(len(users)))

with open('songs.csv', 'r') as fin:
	songs = []
	song_to_index = {}
	song_to_genre = {}
	for song_index,line in enumerate(fin):
		if (song_index) != 0:
			thisLine = line.split(",")
			songs.append(thisLine[0])
			song_to_index[thisLine[0]] = song_index - 1
			song_to_genre[thisLine[0]] = thisLine[2]
	fin.close()
	print("NUM_ITEM" + str(len(songs)))

#not here because need to redistribute with users with greater than 2 songs. if u did it here can't know size for the number of songs each user has
with open('train.csv', 'r') as fin:
	for idx,line in enumerate(fin):
		if idx != 0:
			thisLine = line.split(",")
			if thisLine[0] in user_to_index and thisLine[1] in song_to_index and thisLine[5]:	
				user_index = user_to_index[thisLine[0]]
				song_index = song_to_index[thisLine[1]]
				user_song_dic[user_index].add(song_index)
	fin.close()


# with open('music/test.csv', 'r') as fin:
# 	pos_user_song_test_dic = {}
# 	for idx,line in enumerate(fin):
# 		thisLine = line.split(",")
# 		if idx != 0:
# 			if thisLine[1] in user_to_index and thisLine[2] in song_to_index:
# 				user_index = user_to_index[thisLine[1]]
# 				song_index = song_to_index[thisLine[2]]
# 				pos_user_song_test_dic[user_index] = Set([song_index])
# 	fin.close()


next_user_id = 0
train_structured_arr = np.zeros(7377416, dtype=[('user_id', np.int32), ('item_id', np.int32)])
val_structured_arr = np.zeros(29164, dtype=[('user_id', np.int32), ('item_id', np.int32)]) 
test_structured_arr = np.zeros(29164, dtype=[('user_id', np.int32), ('item_id', np.int32)])
users_with_more_than_2_songs =0
interaction_ind = 0
for user, songs in user_song_dic.iteritems():
	for ind, song_idx in enumerate(list(songs)):
		if ind == 0 and len(songs) > 2: 
		    val_structured_arr[users_with_more_than_2_songs] = (user, song_idx)
		    users_with_more_than_2_songs += 1
		elif ind == 1 and len(songs) > 2:
		    test_structured_arr[users_with_more_than_2_songs] = (user, song_idx)
		else:
		    train_structured_arr[interaction_ind] = (user, song_idx)
		    interaction_ind += 1

with open('train_structured_arr.p','wb') as f:
    pickle.dump(train_structured_arr,f)
with open('val_structured_arr.p','wb') as f:
    pickle.dump(val_structured_arr,f)
with open('test_structured_arr.p','wb') as f:
    pickle.dump(test_structured_arr,f)
with open('song_to_genre.p','wb') as f:
    pickle.dump(song_to_genre,f)



