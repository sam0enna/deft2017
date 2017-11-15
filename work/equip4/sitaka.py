import treetaggerwrapper
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import pandas as pd

class sitaka:

    def  __init__(self):
        self.tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr',TAGDIR='../treetagger/',TAGINENC='utf-8',TAGOUTENC='utf-8')
        #self.emo_lex =
        self.word_lex = pd.read_csv("intern_ressrc/FEEL.csv", sep = ";")
        #self.word2vec = 
        self.bow = ['x1f60d','️','magnifique','x1f389','x1f1f5','omg','x1f4aa','fière','coeur','porcro',"j'aimerai",'xfactor','rumba','folie','parfait',"l'adore",'talk','x263a','gentillesse',"l'access",'adoré','alizée','déchire','presta','danser','accro',"j'aimerais",'rabiot','mignonne','x1f600','épisodes','plait','pda','téléspectateurs','humour',"l'épisode",'*0*','x1f3fd','teamcandiflow','x270c','flow','dvd','desportes','pétillante',"d'émotion",'frissons','félicitation','chapeau','ouiiiii','transporteur']
        self.bonw = ['triste','nul','voulez','x1f44e',"n'existe",'gueant','medias',"l'autre",'faux','logique','pitié','capucine','prétexte','arabes','peuples','voient','bouche','leçons','arrêtez','fdp',"d'la",'diplome','anti','x1f62d','sale','ridicule','savais','soutenu',"n'oubliez",'carnage','principal','ha','xfe332','cons','foutre','utile','francais','qq','mecs','mensonge','media',"qu'hollande",'ménage','profond','2/2','libye','fdg','etre','bonnet','nathalie']
        self.bowm = ['merah',"l'upr",'❓','alexandra','flanby',"d'assister",'prédis','tombera','redeviendra','jesta','bounga','avaient','mohammed','soyons','yann','x1f61a','bonus','sapin','lunettes','perds','balek','fernando','punir','valide','déterminé','amie','taf','consentie','différent','x1f604','topchef','jdis','limites','laïque','com’','mickael','voila','2013.','stp','jsavais','cauchemarencuisine','fiouh','suspendu','esa','commencera','nkm.en','maîtresses',"l'évidence",'petrole','dfaire']
        self.bowo = ['vladimir','actu','afp','actualités','objective','avions','conséquences','forces','quarts','khorasan','n’ai','exclusive','limogés','saint-nazaire','renaud','pentagone','yahoo','negative','n’y','benghalem','juin','serrer','•','angela','salim','@','éviter','«le','bale','single','revue','passation','alerte','province','consultante','ebp-pro','itele','senegal','rfi','origines','liam','dont','sénat','enterrent','mardi','recep','prokurde','gallois','frère','nager']
        self.pos_polarity_lex = list(self.polarity_lex("positive")["word"])
        self.neg_polarity_lex = list(self.polarity_lex("negative")["word"])


    def correction(self, word):
        pass

    def polarity_lex(self, polarity):
        return self.word_lex[(self.word_lex["polarity"] == polarity)]

    def pos_polarity(self, lemmes):
        nb = 0
        for lemme in lemmes:
            if lemme in self.pos_polarity_lex:
                nb = nb +1
        return nb

    def neg_polarity(self, lemmes):
        nb = 0
        for lemme in lemmes:
            if lemme in self.neg_polarity_lex:
                nb = nb + 1
        return nb

    def normalize(self, data):
        data = data.lower()

        stop = set(stopwords.words('french'))
        tokens = word_tokenize(data, 'french')
        tokens = [w for w in tokens if w not in stop]
        return tokens

    def bow_features(self, tokens):
        bow = []
        for word in self.bow:
            if word in tokens:
                bow.append(1)
            else:
                bow.append(0)
        return bow    
    
    def bonw_features(self, tokens):
        bonw = []
        for word in self.bonw:
            if word in tokens:
                bonw.append(1)
            else:
                bonw.append(0)
        return bonw 

    def bowm_features(self, tokens):
        bowm = []
        for word in self.bowm:
            if word in tokens:
                bowm.append(1)
            else:
                bowm.append(0)
        return bowm    
    
    def bowo_features(self, tokens):
        bowo = []
        for word in self.bowo:
            if word in tokens:
                bowo.append(1)
            else:
                bowo.append(0)
        return bowo 

    def polarity(self, nb_bow, nb_bonw):
        if nb_bow > nb_bonw:
            return 1.0 - nb_bonw/nb_bow
        elif nb_bow < nb_bonw:
            return nb_bow/nb_bonw - 1.0
        return 0.0

    def tag(self, data):
        text_tag = self.tagger.tag_text(data)
        text_tag_tuple = treetaggerwrapper.make_tags(text_tag, exclude_nottags=True)
        return [{'word':word, 'pos':pos, 'lemme': lemme} for (word, pos, lemme) in text_tag_tuple]

    def findall(self,rgx, data):
        result = []
        for w in data:
            if w["pos"]==rgx:
                result.append(w)
        return result

    def lemmes_tokens(self, tokens):
        result = []
        for w in tokens:
           result.append(w["lemme"])
        return result

    def number_of_nouns(self, data):
        return len(self.findall("NOM",data))

    def number_of_sym(self, data):
        return len(self.findall("SYM",data))

    def number_of_prp(self, data):
        return len(self.findall("PRP",data))

    def number_of_adj(self, data):
        return len(self.findall("ADJ",data))

    def number_of_adv(self, data):
        return len(self.findall("ADV",data))

    def number_of_abr(self, data):
        return len(self.findall("ABR",data))

    def number_of_int(self, data):
        return len(self.findall("INT",data))

    def number_of_pun(self, data):
        return len(self.findall("PUN",data))

    def nb_syntactic_features(self,data):
        pos = []
        pos.append(self.number_of_sym(data))
        #pos.append(self.number_of_prp(data))
        pos.append(self.number_of_nouns(data))
        pos.append(self.number_of_adj(data))
        pos.append(self.number_of_adv(data))
        pos.append(self.number_of_int(data))
        pos.append(self.number_of_abr(data))
        return pos


