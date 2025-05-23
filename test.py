import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the training data (you can add more items)
train_data = [
    # Food items
    ("apple", "food"),
    ("banana", "food"),
    ("orange", "food"),
    ("rice", "food"),
    ("pasta", "food"),
    ("bread", "food"),
    ("chicken", "food"),
    ("beef", "food"),
    ("milk", "food"),
    ("cheese", "food"),
    ("lettuce", "food"),
    ("tomato", "food"),
    ("onion", "food"),
    ("carrot", "food"),
    ("spinach", "food"),
    ("salmon", "food"),
    ("shrimp", "food"),
    ("watermelon", "food"),
    ("strawberry", "food"),
    ("blueberry", "food"),
    ("sushi", "food"),
    ("pancake", "food"),
    ("cereal", "food"),
    ("grapes", "food"),
    ("apple pie", "food"),
    ("chocolate cake", "food"),
    ("coffee", "food"),
    ("tea", "food"),
    ("potato", "food"),
    ("avocado", "food"),
    ("mango", "food"),
    ("hummus", "food"),
    ("olive oil", "food"),
    ("vinegar", "food"),
    ("water", "food"),
    ("peanut butter", "food"),
    ("jam", "food"),
    ("honey", "food"),
    ("yogurt", "food"),
    ("smoothie", "food"),
    ("pita bread", "food"),
    ("pudding", "food"),
    ("butter", "food"),
    ("cucumber", "food"),
    ("zucchini", "food"),
    ("paprika", "food"),
    ("bacon", "food"),
    ("steak", "food"),
    ("turkey", "food"),
    ("lobster", "food"),
    ("clam", "food"),
    ("oysters", "food"),
    ("taco", "food"),
    ("pizza", "food"),
    ("salsa", "food"),
    ("guacamole", "food"),
    ("sandwich", "food"),
    ("hot dog", "food"),
    ("cheeseburger", "food"),
    ("fried chicken", "food"),
    ("lasagna", "food"),
    ("spaghetti", "food"),
    ("chili", "food"),
    ("soup", "food"),
    ("salad", "food"),
    ("popcorn", "food"),
    ("chips", "food"),
    ("pretzels", "food"),
    ("cake", "food"),
    ("cookies", "food"),
    ("pie", "food"),
    ("french fries", "food"),
    ("ice cream", "food"),
    ("popsicle", "food"),
    ("donut", "food"),
    ("waffle", "food"),
    ("croissant", "food"),
    ("bagel", "food"),
    ("muffin", "food"),
    ("scone", "food"),
    ("granola bar", "food"),
    ("cheese pizza", "food"),
    ("fried rice", "food"),
    ("pad thai", "food"),
    ("ramen", "food"),
    ("dumplings", "food"),
    ("sashimi", "food"),
    ("curry", "food"),
    ("burrito", "food"),
    ("fried noodles", "food"),
    ("quesadilla", "food"),
    ("salmon sushi", "food"),
    ("rice pudding", "food"),
    ("tiramisu", "food"),
    ("baklava", "food"),

    # Non-food items
    ("car", "not food"),
    ("television", "not food"),
    ("laptop", "not food"),
    ("phone", "not food"),
    ("detergent", "not food"),
    ("soap", "not food"),
    ("shampoo", "not food"),
    ("bottle", "not food"),
    ("scissors", "not food"),
    ("toothbrush", "not food"),
    ("pen", "not food"),
    ("notebook", "not food"),
    ("clock", "not food"),
    ("lamp", "not food"),
    ("chair", "not food"),
    ("desk", "not food"),
    ("table", "not food"),
    ("computer", "not food"),
    ("printer", "not food"),
    ("microwave", "not food"),
    ("socks", "not food"),
    ("shoes", "not food"),
    ("jacket", "not food"),
    ("book", "not food"),
    ("glove", "not food"),
    ("hat", "not food"),
    ("fan", "not food"),
    ("earphones", "not food"),
    ("pencil", "not food"),
    ("bed", "not food"),
    ("toys", "not food"),
    ("screwdriver", "not food"),
    ("pliers", "not food"),
    ("wrench", "not food"),
    ("drill", "not food"),
    ("hammer", "not food"),
    ("paint", "not food"),
    ("nail", "not food"),
    ("glue", "not food"),
    ("plastic", "not food"),
    ("cardboard", "not food"),
    ("metal", "not food"),
    ("wood", "not food"),
    ("screw", "not food"),
    ("tape", "not food"),
    ("marker", "not food"),
    ("folder", "not food"),
    ("phone case", "not food"),
    ("headphones", "not food"),
    ("monitor", "not food"),
    ("screen", "not food"),
    ("charger", "not food"),
    ("keyboard", "not food"),
    ("mouse", "not food"),
    ("router", "not food"),
    ("bicycle", "not food"),
    ("helmet", "not food"),
    ("backpack", "not food"),
    ("glasses", "not food"),
    ("wallet", "not food"),
    ("briefcase", "not food"),
    ("gloves", "not food"),
    ("umbrella", "not food"),
    ("trolley", "not food"),
    ("suitcase", "not food"),
    ("shovel", "not food"),
    ("wheelbarrow", "not food"),
    ("spray bottle", "not food"),
    ("mop", "not food"),
    ("broom", "not food"),
    ("vacuum", "not food"),
    ("toilet paper", "not food"),
    ("brush", "not food"),
    ("mirror", "not food"),
    ("toilet", "not food"),
    ("shower", "not food"),
    ("furniture", "not food"),
    ("air conditioner", "not food"),
    ("heater", "not food"),
    ("light bulb", "not food"),
    ("candle", "not food"),
    ("tissue", "not food"),
    ("band-aid", "not food"),
    ("first aid kit", "not food")
]


# Split the data into features (X) and labels (y)
texts, labels = zip(*train_data)

# Create the Naive Bayes classifier with a CountVectorizer
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(texts, labels)
text = """Walmart
Save money. Live better.
( 330) 339 - 3991
MANAGER DIANA EARNEST
231 BLUEBELL DR SW
NEW PHILADELPHIA OH 44663
ST# 02115 OP# 009044 TE# 44 TR#
01301
PET TOY
004747571658
1.97 X
FLOPPY PUPPY
004747514846
1.97 X
SSSUPREME S
070060332153
4.97
2.5 SQUEAK
084699803238
5.92
MUNCHY DMBEL
068113108796
3.77
DOG TREAT
007119013654
2.92
X
X
PED PCH 1
002310011802
0.50
PED PCH 1
002310011802
0.50
X
COUPON 23100
052310037000
1.00-0
HNYMD SMORES
088491226837
F
3.98
FRENCH DRSNG
004132100655 F
1.98 0
3 ORANGES
001466835001 F
5.47 N
BABY CARROTS
003338366602 I
1.48
N
COLLARDS
000000004614KI
1.24
N
CALZONE
005208362080 F
2.50 0
MM RVW MNT
003399105848
19.77
STKOBRLPLABL
001558679414
1.97
X
STKOBRLPLABL
001558679414
1.97
STKO SUNFLWR
001558679410
0.97
X
STKO SUNFLWR
001558679410
0.97
STKO SUNFLWR
001558679410
0.97
STKO SUNFLWR
001558679410
0.97
BLING BEADS
076594060699
0.97
X
X
X
GREAT VALUE
007874203191 F
9.97
LIPTON
001200011224 F
4.48
DRY DOG
002310011035
12.44
SUBTOTAL
93.62
TAX 1
6.750 %
4.59
TOTAL
98.21
VISA
TEND
98.21
US DEBIT
******** **** 9166
I
APPROVAL # 572868
REF # 720900544961
TRANS ID - 387209239650894
VALIDATION - 87HS
PAYMENT SERVICE - E
AID A0000000980840
TC 51319CA81DC22BC7
TERMINAL # SC010764
*Signature Verified
07/28/17
02:39:48
CHANGE DUE
# ITEMS SOLD 25
TC# 0443
0223 1059 8001 5140
0.00
Low Prices You Can Trust. Every Day.
Y
07/28/17
02:39:48
***CUSTOMER COPY***"""
text = """Walmart
Save money. Live better.
( 330) 339 - 3991
MANAGER DIANA EARNEST
231 BLUEBELL DR SW
NEW PHILADELPHIA OH 44663
ST# 02115 OP# 009044 TE# 44 TR#
01301
PET TOY
004747571658
1.97 X
FLOPPY PUPPY
004747514846
1.97 X
SSSUPREME S
070060332153
4.97
2.5 SQUEAK
084699803238
5.92
MUNCHY DMBEL
068113108796
3.77
DOG TREAT
007119013654
2.92
X
X
PED PCH 1
002310011802
0.50
PED PCH 1
002310011802
0.50
X
COUPON 23100
052310037000
1.00-0
HNYMD SMORES
088491226837
F
3.98
FRENCH DRSNG
004132100655 F
1.98 0
3 ORANGES
001466835001 F
5.47 N
BABY CARROTS
003338366602 I
1.48
N
COLLARDS
000000004614KI
1.24
N
CALZONE
005208362080 F
2.50 0
MM RVW MNT
003399105848
19.77
STKOBRLPLABL
001558679414
1.97
X
STKOBRLPLABL
001558679414
1.97
STKO SUNFLWR
001558679410
0.97
X
STKO SUNFLWR
001558679410
0.97
STKO SUNFLWR
001558679410
0.97
STKO SUNFLWR
001558679410
0.97
BLING BEADS
076594060699
0.97
X
X
X
GREAT VALUE
007874203191 F
9.97
LIPTON
001200011224 F
4.48
DRY DOG
002310011035
12.44
SUBTOTAL
93.62
TAX 1
6.750 %
4.59
TOTAL
98.21
VISA
TEND
98.21
US DEBIT
******** **** 9166
I
APPROVAL # 572868
REF # 720900544961
TRANS ID - 387209239650894
VALIDATION - 87HS
PAYMENT SERVICE - E
AID A0000000980840
TC 51319CA81DC22BC7
TERMINAL # SC010764
*Signature Verified
07/28/17
02:39:48
CHANGE DUE
# ITEMS SOLD 25
TC# 0443
0223 1059 8001 5140
0.00
Low Prices You Can Trust. Every Day.
Y
07/28/17
02:39:48
***CUSTOMER COPY***"""
test_items=[]
lines = text.split("\n")
for i in lines:
    try:
        int(i)
    except:
        test_items.append(i)
#Test the classifier with some sample items
#test_items = ["bannana","Dodge","ORNG","sour cream","Bag of rice", "Television", "Socks", "Mango", "Computer", "Pasta sauce", "Shoes", "Apple"]
predictions = model.predict(test_items)

# Output the results
for item, prediction in zip(test_items, predictions):
    print(f"{item}: {prediction}")
    
# Optional: Evaluate model accuracy
# Splitting data into training and test sets for evaluation
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Re-train on the training data and test on the test data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nAccuracy on the test set:", accuracy_score(y_test, y_pred))
