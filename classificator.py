from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.corpus import stopwords 


with open('lda.lda', 'rb') as file:
	lda = pickle.load(file)

with open('vect.vect', 'rb') as file:
	vect = pickle.load(file)

stop = set(stopwords.words('russian'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    result = ''.join([i for i in doc if not i.isdigit()])
    stop_free = " ".join([i for i in result.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    
    return normalized

def get_topics(text):
	themes = {0: 'карты',
	1: 'страховка',
	2: 'неизвестно',
	3: 'обслуживание',
	4: 'условия договора',
	5:	'обслуживание',
	6: 'условия договора',
	7: 'обслуживание',
	8: 'условия акций',
	9: 'время обслуживания',
	10: 'обслуживание',
	11:	'неизвестно',
	12:	'условия ипотеки',
	13:	'штрафы',
	14:	'онлайн-обслуживание'
	}
	cleaned = clean(text)
	tf = vect.transform([cleaned])
	predict = lda.transform(tf)
	tmp = [(index, predict[0][index]) for index in range(len(predict[0]))]
	res = {}
	for item in tmp:
		if themes[item[0]] in res.keys():
			res[themes[item[0]]] += item[1]
		else:
			res[themes[item[0]]] = item[1]
	result = []
	for key in res.keys():
		value = res[key]
		temp = [key,value]
		result.append(temp)
	return(sorted(result, key=lambda x: -x[1])[0][0])

vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
cls = pickle.load(open("cls.pickle", "rb"))
    

def get_tone(text):

    sample = vectorizer.transform([text])
    return(cls.predict_proba(sample)[0][0])

if __name__ == '__main__':
	text = """
Ну конечно, какую еще оценку можно поставить Сбербанк -1.
Первое, с чем я столкнулась при сотрудничестве, что мой паспорт при оформлении кредита сотрудник отправил своей знакомой. 
Я оставила без наказано, продолжив сотрудничать так объект был аккредитован только этим ГОВНО банком.
Теперь я столкнулась с ситуацией, когда с моей карты списали дважды сумму в размере 9000 за одну покупку.
Карл, это дело разбирают уже 20 дней...что можно разбирать, обращение 180 407 0080 193800 осталось без внимания...звонишь сотрудники мычат в трубку...
Аналогичная ситуация случилась с картой другого банка, вопрос решился за два дня и деньги вернулись сразу же.
Мне сегодня срочно нужно вернуть деньги...а они рассматривают..звонишь просишь поставить в приоритет..игнор!
А знаете какая у меня мечта!? Как только достоится дом, сразу же рефинансироваться и НИКОГДА никогда в жизни не иметь дело с СБЕРБАНК РОССИИ!!!!!!
"""
	print(get_topics(text))
	print(get_tone(text))
