from assign1 import *




if __name__ == "__main__":
    gold = pd.read_csv('gold_test.csv')
    print(gold.ratings.tolist())
    val=main("train.csv","test.csv")
    print(val)
