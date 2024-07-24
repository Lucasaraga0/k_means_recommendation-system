import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans


def user_based_kmeans(data: pd.DataFrame, user_id: int, number_of_recommendations = 10):
    """
    essa abordagem usa apenas a tabela de ratings para recomendar itens ainda nao vistos a um usuario  
    """
    user_data = pd.DataFrame([data.iloc[user_id]]) 
    user_evaluations = []
    
    for item in user_data:
        value = user_data.at[user_id, item]
        if value != 0:
            user_evaluations.append([value, item])        
    test_size = np.ceil(len(user_evaluations)) * 0.3
    user_test_data = []
    
    for i, item in enumerate(user_evaluations):
        if i > test_size:
            break
        elif item[0] > 3:
            user_test_data.append(item)
            user_data.at[user_id, item[1]] = 0
    
    alt_data = data.drop(index = user_id)
    
    model = KMeans(n_init=10)
    model.fit(alt_data)
    clusters = model.predict(alt_data)
    alt_data['clusters'] = clusters.tolist()
    user_cluster = model.predict(user_data)
    cluster_value = user_cluster[0]
    user_data['clusters'] = user_cluster.tolist()
    alt_data = pd.concat([alt_data, user_data])
    alt_data = alt_data.loc[alt_data['clusters'] == cluster_value]
    items_means = []
    
    for item in alt_data:
        if alt_data.at[user_id, item] == 0:
            temp_data = alt_data.copy()
            temp_data = temp_data[temp_data[item] != 0]
            mean = temp_data[item].mean()
            items_means.append([mean, item])    
    
    items_means.sort(reverse= True)
    comparison_list = []
    
    for valor_previsto, item1 in items_means:
        for valor_real,item2 in user_test_data:
            if item1 == item2:
                comparison_list.append((valor_real - valor_previsto)**2 )
    comparison_list = np.array(comparison_list)
    comparison_list = comparison_list[~np.isnan(comparison_list)]
    rmse = np.sqrt(np.sum(comparison_list)/len(comparison_list))
    
    return items_means[0:number_of_recommendations], rmse

def return_user_based_recommendation(data, user_id, number_of_recommendations = 10):
    recomended_items, _ = user_based_kmeans(data, user_id, number_of_recommendations)

    def pegar_segundo_item(lista):
        lista_aux = []
        for item in lista:
            lista_aux.append(item[1])
    # print(lista_aux)
        return lista_aux
    
    return pegar_segundo_item(recomended_items)


def get_liked_items(rating_data, user_index, return_rated = True):

    """
    retorna os itens que foram bem avaliados pelo usuarios 
    """

    #primeiro pegar os itens ja avaliados pelos usuario
    rated_items = []
    liked_items = []

    for item in rating_data:
        valor = rating_data.at[user_index, item]
        if valor != 0 and item != 'User Average Rating':
            rated_items.append([valor, item])
            if valor >= 4:
                liked_items.append([valor, item]) 
    if return_rated:
        return liked_items, rated_items
    return liked_items

def item_based_kmeans(item_characteristics: pd.DataFrame, items_columns, rated_items, liked_items, number_of_recommendations = 10, avg_rating_column:str = 'Movie Average Rating'):
    
    """
    essa abordagem se utiliza das caracteristicas dos itens bem avaliados por um usuario para conseguir indicar itens bem avaliados entre
    aqueles que sao similares a estes.
    """

    # rated_items sao todos aqueles itens que o usuario ja avaliou, rated items pode ter so o nome
    # liked_items sao os n mais bem avaliados do usuario usuario, precisa conter a linha inteira para ter as caracteristicas sobre o item, somente o nome nao vai ser suficiente

    removed_items = []

    for item in items_columns:
        if item in rated_items:
            removed_items.append(item)
    
    aux_data = item_characteristics.select_dtypes(exclude = ['object'])
    aux_data.drop(columns = removed_items, inplace = True)
    model = KMeans()
    model.fit(aux_data)
    aux_data['clusters'] = model.predict(aux_data)

    rec_items = []
    movies_names = item_characteristics['Movie Name']
    #pegar o cluster de cada um dos itens que o usuario gostou, dentre os desse cluster pegar aqueles com maior media (movie_average_rating da tabela de movie_ratings)
    for item in liked_items:
        #print(item)
        item = item_characteristics.loc[item_characteristics['Movie Name'] == item[1]]
        item = item.select_dtypes(exclude = ['object'])
        cluster = model.predict(item)
        #print(int(cluster))
        temp_data = aux_data.copy()
        
        #print(cluster)
        cluster = cluster[0]
        temp_data = temp_data.loc[temp_data['clusters'] == int(cluster)]
        
        temp_data.sort_values(by = avg_rating_column, ascending= False, inplace= True)
        best = temp_data[avg_rating_column][0:number_of_recommendations*2]
        #temp_data['Movie Name'] = \

        #print(temp_data.head())
        correct_movies_names = movies_names.reindex(temp_data.index)
        temp_data['Movie Name'] = correct_movies_names
        #print(best)
        #print(type(best))
        best = temp_data['Movie Name'][0: number_of_recommendations*2]
        rec_items.append(best)
    
    #depois disso ja se tem os itens mais recomendados para cada item, agora eh fazer uma contagem e retornar aqueles que mais aparecem ou aqueles com a maior nota?
    #print(rec_items)
    rec_items = pd.concat(rec_items)
    #print(rec_items)
    counter = Counter(rec_items)
    final_rec = counter.most_common(number_of_recommendations)
    return [item for item, count in final_rec]

def popularity_based_recommendation(rating_data, rated_items, item_based_items = [], user_based_items = [], number_of_recommendations =10):
    
    usables = []
    
    for item in rating_data:
        if item not in rated_items and item not in item_based_items and item not in user_based_items and item != 'User Average Rating':
            usables.append(item)

    usable_data = rating_data[usables]
    zeros = (usable_data == 0).sum()
    zeros = zeros.sort_values()
    
    popular = zeros.index[:number_of_recommendations].tolist()
    return popular



def recommendation_system(user_index, rating_data, items_characteriscts, items, n_user = 6, n_items = 4, n_popularity = 4):
    print('Buscando as melhores recomendacoes para você: \n')
    # para o user based eh basicamente so aplciar a funcao 
    user_based_items = return_user_based_recommendation(data= rating_data, user_id= user_index, number_of_recommendations= n_user)
    
    print('Usuarios como voces gostaram de :')    
    for item in user_based_items:
        print(item)

    #para o item based, vai ser preciso pegar os itens que o usuario ja avaliou e aqueles que ele mais gostou 
    liked_items, rated_items = get_liked_items(rating_data= rating_data, user_index= user_index)
    
    #pegar os itens do item_based
    item_based_items =  item_based_kmeans(item_characteristics= items_characteriscts, items_columns= items, rated_items= rated_items, liked_items= liked_items, number_of_recommendations= n_items)
    print('\nBaseado em suas ultimas avaliacoes: ')    
    for item in item_based_items:
        print(item)
    
    
    #por fim os itens por popularidade
    popularity_based = popularity_based_recommendation(rating_data= rating_data, rated_items= rated_items, item_based_items= item_based_items, user_based_items = user_based_items, number_of_recommendations = n_popularity)

    print('\nEm alta: ')    
    for item in popularity_based:
        print(item)

    recommended_items = user_based_items + item_based_items + popularity_based
    

    return recommended_items