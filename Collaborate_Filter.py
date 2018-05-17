import math

class Collaborate_Filter:
    def __init__(self, input_file, k):
        self.dataset = input_file
        self.uu_dataset ,self.ii_dataset=self.load_data(self.dataset)
        self.k = k
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                             Pearson Correlation                              """
    """                                                                              """    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def pearson_correlation(self, user1, user2):
        result = 0.0
        try:
            user1_data = self.uu_dataset[user1]
            user2_data = self.uu_dataset[user2]
        except:
            return 0
        rx_avg = self.user_average_rating(user1_data)
        ry_avg = self.user_average_rating(user2_data)
        sxy = self.common_items(user1_data, user2_data)

        top_result = 0.0
        bottom_left_result = 0.0
        bottom_right_result = 0.0
        for item in sxy:
            rxs = user1_data[item]
            rys = user2_data[item]
            top_result += (rxs - rx_avg)*(rys - ry_avg)
            bottom_left_result += pow((rxs - rx_avg), 2)
            bottom_right_result += pow((rys - ry_avg), 2)
        bottom_left_result = math.sqrt(bottom_left_result)
        bottom_right_result = math.sqrt(bottom_right_result)
        
        if bottom_left_result*bottom_right_result!=0:
            result = top_result/(bottom_left_result * bottom_right_result)
            return result
        else:
            return 0

    def user_average_rating(self, user_data):
        avg_rating = 0.0
        size = len(user_data)
        for (movie, rating) in user_data.items():
            avg_rating += float(rating)
        avg_rating /= size * 1.0
        return avg_rating

    def common_items(self, user1_data, user2_data):
        result = []
        ht = {}
        for (movie, rating) in user1_data.items():
            ht.setdefault(movie, 0)
            ht[movie] += 1
        for (movie, rating) in user2_data.items():
            ht.setdefault(movie, 0)
            ht[movie] += 1
        for (k, v) in ht.items():
            if v == 2:
                result.append(k)
        return result

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                             K Nearest Neighbors                              """
    """                                                                              """    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def k_nearest_neighbors(self, user, k):
        neighbors = []
        result = []
        for (user_id, data) in self.uu_dataset.items():
            if user_id == user:
                continue
            upc = self.pearson_correlation(user, user_id)
            neighbors.append([user_id, upc])
        sorted_neighbors = sorted(neighbors, key=lambda neighbors: (neighbors[1], neighbors[0]), reverse=True)
        for i in range(k):
            if i >= len(sorted_neighbors):
                break
            result.append(sorted_neighbors[i])
        return result

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                  Predict                                     """
    """                                                                              """    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def predict(self, user, item, k_nearest_neighbors):
        valid_neighbors = self.check_neighbors_validattion(item, k_nearest_neighbors)
        if not len(valid_neighbors):
            return 0.0
        top_result = 0.0
        bottom_result = 0.0
        for neighbor in valid_neighbors:
            neighbor_id = neighbor[0]
            neighbor_similarity = neighbor[1]   # Wi1
            rating = self.uu_dataset[neighbor_id][item] # rating i,item
            top_result += neighbor_similarity * rating
            bottom_result += neighbor_similarity
        if(bottom_result==0):
            return 0
        result = top_result/bottom_result
        return result

    def check_neighbors_validattion(self, item, k_nearest_neighbors):
        result = []
        for neighbor in k_nearest_neighbors:
            neighbor_id = neighbor[0]
            # print item
            if item in self.uu_dataset[neighbor_id].keys():
                result.append(neighbor)
        return result

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                             Helper Functions                                 """
    """                                                                              """    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def load_data(self, data_set):
        """
        load data and return three outputs for extention purpose
        only one output is enough in practice (uu_dataset)

        """
        uu_dataset = {}
        ii_dataset = {}
        for row in data_set:

            """
            user-user dataset: [0: Movie Name  1: Rating]

            """
            uu_dataset.setdefault(row[0], {})
            uu_dataset[row[0]].setdefault(row[2], float(row[1]))

            """
            item-item dataset: [0: user id  1: Rating]

            """
            ii_dataset.setdefault(row[2], {})
            ii_dataset[row[2]].setdefault(row[0], float(row[1]))
        return uu_dataset, ii_dataset
    
    def display(self, k_nearest_neighbors, prediction):
        for neighbor in k_nearest_neighbors:
            print(neighbor[0], neighbor[1])
        print("\n")
        print(prediction)

    def quit(self, err_desc):
        tips = "\n" + "TIPS: " + "\n"   \
                + "--------------------------------------------------------" + "\n" \
                + "Pragram name: lingzhe_teng_collabFilter.py" + "\n" \
                + "First parameter: Input File, e.g. ratings-dataset.tsv" + "\n" \
                + "Second parameter:  User ID, e.g. Kluver" + "\n" \
                + "Thrid parameter:  Movie, e.g. The Fugitive" + "\n" \
                + "Fourth parameter: K, e.g. 10" + "\n" \
                + "--------------------------------------------------------" + "\n" \
                + "Note:" + "\n" \
                + "Please use double quotation marks, such as \"USER\'S ID\" or \"MOVIEW\'S NAME\", for User ID and Moview parameters" + "\n" 


        raise SystemExit('\n'+ "PROGRAM EXIT: " + err_desc + ', please check your input' + '\n' + tips)


