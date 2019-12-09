# Testing of components, languages and domains of the Red Alert API 
# Insikt Intelligence S.L. 2019

import requests
import json
import pandas as pd
import colorama
from colorama import Fore, Style

BASE_URL = 'http://127.0.0.1:5000'

header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}

#___________________________________________________________________________________________________
print( Fore.BLUE+"Test /vectorize")
print(Style.RESET_ALL)
data = {'text':'English is a West Germanic language that was first spoken in early medieval England and eventually became a global lingua franca.[4][5] It is named after the Angles, one of the Germanic tribes that migrated to the area of Great Britain that later took their name, as England.'}
datajson = json.dumps(data)
response = requests.post("{}/vectorize".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

data = {'text':'El español o castellano es una lengua romance procedente del latín hablado. Pertenece al grupo ibérico y es originaria de Castilla, reino medieval de la península ibérica. Se conoce también por el americanismo coloquial de castilla (por ejemplo: «hablar castilla», «entender castilla»),nota 1​32​33​ común en algunas áreas rurales e indígenas entre México, Perú y la Patagonia,34​ pues el castellano se empezó a enseñar poco después de la incorporación de los nuevos territorios a la Corona de Castilla'}
datajson = json.dumps(data)
response = requests.post("{}/vectorize".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

data = {'text':"Le français est une langue indo-européenne de la famille des langues romanes. Le français s'est formé en France (variété de la « langue d’oïl , qui est la langue de la partie septentrionale du pays)."}
datajson = json.dumps(data)
response = requests.post("{}/vectorize".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

data = {'text':'شجرة خضراء مبنى عال  رجل مُسن جدا البيت القديم ألاحمر صديق لطيف جدا أقرأ كتابا احيانا أنا لن أدخّن ابدا هل أنت وحدك؟ انه سعيد انها سعيدةانه امريكى الجنسية '}
datajson = json.dumps(data)
response = requests.post("{}/vectorize".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

data = {'text':'Toate fiintele umane se nasc libere si egale în demnitate si în drepturi. Ele sunt înzestrate cu ratiune si constiintã, si trebuie sã se comporte unele fatã de altele în spiritul fraternitãtii.'}
datajson = json.dumps(data)
response = requests.post("{}/vectorize".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

#___________________________________________________________________________________________________
print(Fore.BLUE+"Test /probability Jihadist - Positive")
print(Style.RESET_ALL)

data = {'text': 'Allahum decrease the #fitnah between #Muslims and give patience to our #Muslim #lions who are fighting for the Deen', 'lang':'en','classifier':'Jihadist-English-Insikt.model','user_id':'berta01','case_id':'0053'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] > 0.5:
	print(Fore.GREEN+"PASS in "+data['lang'])
else:
	print(Fore.RED+ "FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text': 'los infieles deben morir en nombre de Ala para recuperar el Califato los leones lucharemos en la guerra y mataremos a todos los occidentales','classifier':'Jihadist-Spanish-Insikt.model','lang':'es'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] > 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text':"les infidèles doivent mourir au nom d'Ala pour récupérer le califat, les lions vont se battre et tuer tous les occidentaux",'lang':'fr','classifier':'all'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] > 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text': '(إذا قال الرجل هلك الناس فهو أهلكهم )  من قال بأن الجهاد قد انتهى فليصحح أفكاره  تذكر دوما (سيجعل الله بعد عسر يسرا)  واستعن بالله ولا تعجز', 'classifier':'Jihadist-Arabic-Insikt.model','lang':'ar'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] > 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text':'creștini de moarte.armata ISIS victorioasă. infidelii trebuie să moară în numele Ala pentru a recupera califatul, leii vor lupta în război și vor ucide toți occidentalii moartea necredincioșilor','lang':'ro','classifier':'Jihadist-Romanian-Insikt.model'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] > 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)
#___________________________________________________________________________________________________
print(Fore.BLUE+"Test /probability Jihadist - Negative")
print(Style.RESET_ALL)

data = {'text': 'I want to go to the beach next holidays and relax and have fun', 'lang':'en','classifier':'Jihadist-English-Insikt.model','user_id':'berta01','case_id':'0053'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] < 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text': 'El español o castellano es una lengua romance procedente del latín hablado. Pertenece al grupo ibérico y es originaria de Castilla, reino medieval de la península ibérica. Se conoce también por el americanismo coloquial de castilla (por ejemplo: «hablar castilla», «entender castilla»),nota 1​32​33​ común en algunas áreas rurales e indígenas entre México, Perú y la Patagonia,34​ pues el castellano se empezó a enseñar poco después de la incorporación de los nuevos territorios a la Corona de Castilla','classifier':'Jihadist-Spanish-Insikt.model','lang':'es'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] < 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text':"J'ai mangé du chocolat. Je suis allé à la piscine ",'lang':'fr','classifier':'all'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] < 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text': 'شجرة خضراء مبنى عال  رجل مُسن جدا البيت القديم ألاحمر صديق لطيف جدا أقرأ كتابا احيانا أنا لن أدخّن ابدا هل أنت وحدك؟ انه سعيد انها سعيدةانه امريكى الجنسية', 'classifier':'Jihadist-Arabic-Insikt.model','lang':'ar'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] < 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text':'Toate fiintele umane se nasc libere si egale în demnitate si în drepturi. Ele sunt înzestrate cu ratiune si constiintã, si trebuie sã se comporte unele fatã de altele în spiritul fraternitãtii.','lang':'ro','classifier':'Jihadist-Romanian-Insikt.model'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] < 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)
#___________________________________________________________________________________________________
print(Fore.BLUE+ "Test /probability ExtremeRight - Positive" )
print(Style.RESET_ALL)

data = {'text': 'Without the White race the world would be unrecognizable today. A White Ethnostate is needed now and forever', 'lang':'en','classifier':'ExtremeRight-English-Insikt.model','user_id':'berta01','case_id':'0053'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] > 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text': 'Los inmigrantes ensucian el pais, estoy arto de tanta mierda de izquierdas y el coletas quelo cuelguen. Viva España Viva Franco','lang':'es','classifier':'all'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] > 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text':"Les problèmes d'immigration en France seront résolus dans trois mois en raison de l'action meurtrière d'Ebola",'lang':'fr','classifier':'ExtremeRight-French-Insikt.model'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] > 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text': 'مهاجرون قذرون من البلاد ، أنا غادر للغاية ، وتعلقها أسلاك التوصيل المصنوعة. يعيش أسبانيا يعيش فرانكو ','lang':'ar', 'classifier':'ExtremeRight-Arabic-Insikt.model'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] > 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text':'El a declarat superioritatea rasei ariene, cu un succes deosebit i-a definit pe evrei ca paraziți. Eliminarea sa a făcut parte din procesul sângeros. Mein Kampf a subliniat necesitatea de a avea Germania sub control nazist - expansiunea militară, eliminarea curselor impure și autoritarismul dictatorial','lang':'ro','classifier':'ExtremeRight-Romanian-Insikt.model'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] > 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

#___________________________________________________________________________________________________
print(Fore.BLUE+"Test /probability ExtremeRight - Negative")
print(Style.RESET_ALL)

data = {'text': 'The Big Bang Theory is an American television sitcom created by Chuck Lorre and Bill Prady, both of whom served as executive producers on the series, along with Steven Molaro.', 'lang':'en','classifier':'ExtremeRight-English-Insikt.model','user_id':'berta01','case_id':'0053'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] < 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text': 'Rosalía nació el 25 de septiembre de 1993 en San Esteban de Sasroviras. Es la segunda hija del matrimonio conformado por José Manuel Vila y Pilar Tobella. Su hermana mayor, llamada Pilar, actualmente trabaja con ella como su estilista. ','classifier':'all','lang':'es'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] < 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text':"J'ai mangé du chocolat. Je suis allé à la piscine ",'lang':'fr','classifier':'ExtremeRight-French-Insikt.model'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] < 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text': 'شجرة خضراء مبنى عال  رجل مُسن جدا البيت القديم ألاحمر صديق لطيف جدا أقرأ كتابا احيانا أنا لن أدخّن ابدا هل أنت وحدك؟ انه سعيد انها سعيدةانه امريكى الجنسية', 'classifier':'ExtremeRight-Arabic-Insikt.model','lang':'ar'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] < 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Foe.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

data = {'text':'Nu vorbesc bine limba română','classifier':'ExtremeRight-Romanian-Insikt.model','lang':'ro'}
datajson = json.dumps(data)
response = requests.post("{}/probability".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())
answer=response.json()
if answer['probability'] < 0.5:
        print(Fore.GREEN+"PASS in "+data['lang'])
else:
        print(Fore.RED+"FAILED in " +data['lang'])
print(Style.RESET_ALL)

#___________________________________________________________________________________________________
print(Fore.BLUE+"Test /analyze")
print(Style.RESET_ALL)

data = {'text': 'He declared the superiority of the Aryan race, with particular success defined the Jews as parasites. Its elimination was part of the bloody process. Mein Kampf stressed the need to have Germany under Nazi control - military expansion, the elimination of impure races and dictatorial authoritarianism', 'lang':'en','classifier':'ExtremeRight-English-Insikt.model','user_id':'berta01','case_id':'0053'}
datajson = json.dumps(data)
response = requests.post("{}/analyze".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

data = {'text': 'El declaró la superioridad de la raza blanca aria, con particular acierto definió los judíos como parásitos. Su eliminación era parte del proceso sangriento. El Mein Kampf remarcó la necesidad de tener a Alemania bajo el control nazi - la expansión militar, la eliminación de razas impuras y el autoritarismo dictatorial','lang':'es','classifier':'all'}
datajson = json.dumps(data)
response = requests.post("{}/analyze".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

data = {'text':"Il déclara la supériorité de la race aryenne, avec un succès particulier, définissant les Juifs comme des parasites. Son élimination faisait partie du processus sanglant. Mein Kampf a insisté sur la nécessité de placer l'Allemagne sous le contrôle des nazis: expansion militaire, élimination des races impures et autoritarisme dictatorial.",'lang':'fr','classifier':'ExtremeRight-French-Insikt.model'}
datajson = json.dumps(data)
response = requests.post("{}/analyze".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

data = {'text': 'أعلن تفوق العرق الآري ، مع نجاح خاص عرف اليهود على أنهم طفيليات. كان القضاء عليه جزءًا من العملية الدموية. وشدد مين Kampf على الحاجة إلى أن تكون ألمانيا تحت السيطرة النازية - التوسع العسكري ، والقضاء على الأعراق غير النبيلة ', 'classifier':'ExtremeRight-Arabic-Insikt.model'}
datajson = json.dumps(data)
response = requests.post("{}/analyze".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

data = {'text':'El a declarat superioritatea rasei ariene, cu un succes deosebit i-a definit pe evrei ca paraziți. Eliminarea sa a făcut parte din procesul sângeros. Mein Kampf a subliniat necesitatea de a avea Germania sub control nazist - expansiunea militară, eliminarea curselor impure și autoritarismul dictatorial','lang':'ro','classifier':'ExtremeRight-Romanian-Insikt.model'}
datajson = json.dumps(data)
response = requests.post("{}/analyze".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())


#___________________________________________________________________________________________________
print(Fore.BLUE+"Test /terms")
print(Style.RESET_ALL)
data_terms={'dataset':["TAFSIR OF SURAH AS SAFFAT (THE ELEVEN SHADES OF KUFR): If you can not see this chirbit  listen to it here http... http://t.co/2BhWTvHqsD","394 : DISEASES OF THE HEART    .  . THE 11 SHADES OF KUFR .  . By Shaikh Abdullah Faisal09.10.2012 (Night Dars)NOTES ty... http://t.co/UlSxcPj6","I downloaded mp3 Diseases of the Heart (2) Kufr  Envy  Hypocrisy and Greed (morning dars 07.22.12).mp3   http://t.co/V4KT8MDF","366  Diseases of the Heart   Kufr  Envy  Hypocrisy and Greed: DISEASES OF THE HEART: Kufr Through Greedby Shaik... http://t.co/onbAQtz6","I downloaded mp3 Kufr In Perspective.mp3   http://t.co/g6uLsWaN","I downloaded mp3 Kufr Bit Taghoot DEBATE   Shaikh Faisal VS. Sofiane.mp3   http://t.co/E9XnFYOw","ANIC is a Tāghūtī organisation founded upon KUFR!! Many are unaware of the reality of ANIC (Australian National Imams Council)  which is run by its president  Suleiman Shady (who permits joining the army  police force and making tahākum). It s an organisation which has abolished the principle of Walā  and Barā   taken democracy as its way of life  whilst engaging in interfaith and much more which will be covered in this brief post Inshā Allāh. Tom Zreika is also the co founder of ANIC who s a kāfir mushrik lawyer  and he wrote an article back in 2006 speaking about the recommendations which the organisation of ANIC strictly upholds. I provided a link to the document in the footnotes which was released just before the formation of the Australian National Imams Council (ANIC) in 2006 entitled  Australian Imams: The Way Forward.  [1].For context  this document was released when the government s discredited  Muslim Reference Group  was discontinued and Sheikh Taj was being ousted as Mufti.Among the clear nullifiers this co founder of ANIC has founded the organisation upon include;Nullifier #1 — “To the fullest extent uphold and promote the rule of law”. ANIC is not just upholding man made Tāghūtī laws  but promoting them and calling upon the muslim community to adhere to these laws and even make tahākum to it..!!Nullifier #2 — “Not in any way  shape or form derogate from the full compliance of the laws of Australia”. Allāhul Musta ān  they have founded ANIC based upon pure compliance with Australian man made laws  without even derogating a single aspect of the law  this is clear major shirk of obedience!Allāh سُحانه وتعالى said those who obey the mushrikīn in permitting the ميتة  you would be mushrikīn  so imagine someone who obeys a complete man made Kufr constitution?!Nullifier #3 — “Not justify the breaking of the laws of Australia”. This implies that if a law prohibited something that s halāl or even wājib in the sharī ah  then ANIC says it will never justify breaking it subhān Allāh! This negates major fundamentals of Walā  and Barā  which we ll come too Inshā Allāh.Nullifier #4 — “Re affirm its allegiance to the Nation of Australia  its symbols and insignia”. Allāhu akbar!! Can you imagine someone giving allegiance to the Nation of Quraysh  Abū Jahl  Abū Lahab and the mushrikīn?! This purely falls under the nullifier of giving loyalty  allegiance and support to the kuffār.Nullifier #5 — “Become a member of an emergency organisation (and promote same) in their locality such as the Bush Fire Service  Surf Life Saving  or any other emergency service provider”. One of the agreements ANIC made with the kuffār is to become members of an emergency organisation and promote it  and this is exactly what the head of ANIC did  the Tāghūt Suleiman Shady (may Allāh curse him)  along with the Muftī  giving fatwās that you can join the army and police force!Nullifier #6 — “Do all things necessary to prevent any radicalisation orthe breeding of fanatical opinions”. Lā hawlā walā quwatā ilā billāh  it s as if we see this right infront of our eyes! Look at how they are being funded by the Tawāghīt to spy upon the muslims  setting up de radicalisation “shuyūkh” in numerous places  to prevent the muslims from believing in what s obligatory within their Dīn of Tawhīd and what it entails of takfīr upon the mushrikīn  while being taught to uphold the laws of Kufr  which is an absolute negator of Walā  and Barā .Nullifier #7 — “Conduct themselves in a manner consistent with Australian values”. Now this is explained by what they previously mentioned of upholding and promoting the Kufr man made law  and ANIC frequently repeats these words in their announcements and letters  wallāhul musta ān.Moreover  those who join this disbelieving organisation are not exempted from blame whatsoever  no matter what excuse they use  such as the most common preventing other sects from banning books and other weak claims  as maslaha for da wah does not permit committing Kufr whatsoever.The noble Imām  Shaykh Sulaymān Ibn Nāsir al  Alwān (فك الله أسره) says:“The Haneef is the one who turns away from Shirk. The Muwahhid Haneef  is he who is determined to worship Allāh and avoids polytheism in all of its different types  forms  shades and colours.He avoids the shirk of the past and present  from worshipping Ahwā  (desires)  Asnām (statues)  Awthān (idols)  and making du ā to other than Allah.He also has to avoid implementing the ruling of the Tāghūt  and making Tahākum to it (i.e. seeking judgement from the Tāghūt)  or joining the disbelieving organizations and what s similar to that.”Concerning the famous Āyah on those who mocked the reciters of the Qur ān  Shaykh al ‘Allāmah ‘Abdul ‘Azīz at Tuwayla ī (تقبله الله) provides outstanding benefits in his amazing commentary upon “Nawāqidh al Islām” (Nullifier #6)  he writes:“And another benefit it contains is that the one who participates in Kufr and being pleased with it  and sitting with the one who says such words in a fashion that accompanies approving of it  (then) ALL of them are Kuffār  for indeed in the verse Allāh has judged the Kufr of every individual within that group of people who sat by & did not give an exception or excuse to any of them  despite the fact that the one speaking is only 1 person and the rest are simply listeners  and as for  the group that Allāh has pardoned   then it has been said it was a man who rebuked some of their speech  and it has been said  what is meant is that a group among them repented  and Allāh pardoned them  and the other group remained upon its Kufr and Hypocrisy  so that is the one which is punished (as stated in the Āyah).”So while the Imāms of Kufr are passing out fatwā on behalf of every single one of your registered names  don t think you will be excused from Kufr for remaining silent  rather it s necessary that you make barā ah from this KUFR organisation in totality!Therefore  I call upon all the members of ANIC to make tawbah to Allāh and publicly make barā ah from this Tāghūt organisation which has harmed muslims in favour of the kuffār!#ANIC | #KUFR | #TĀGHŪT | #MURJIAH","What is the real meaning of Secularism ? What is its definition and purpose ? Are the people who call for or support Secularism in the elections still Muslim ? The definition of Secularism according to the Merriam Webster Dictionary is :   indifference to or rejection or exclusion of religion and religious considerations  To put it simply  a person who calls for Secularism rejects and disbelieves in Allah (SWT)  he is an Atheist when it comes to Politics. Secularism is Atheism in Politics. The ones who adhere and follow this ideology are Murtadeen to the highest order. As Allah (SWT) says in the glorious Qur an : And whosoever does not judge by what Allâh has revealed  such are the Kâfirûn  So the one who judges by man made laws is a Mushrik and a disbeliever  Allah (SWT) says:“Legislation is not but for Allah.” Ibn Kathir (RA) says in Al Bidayyah wa l Nihayyah Book 13  about the Mongols who claimed to be Muslim but still ruled with Al Yaasiq [the laws of the Tartars]:“ Whoever does that  he has disbelieved by the Ijmaa of the Muslims.” [al Bidayyah Wa l Nihayyah  13/118 119] Nobody has the right to Legislate besides Allah (SWT)  and to disbelieve in Allah (SWT)s laws in the matters of state governance and politics is Kufr Al Akbar","We do not eat the meat of the animal Slaughtered by a Rafidhi   Because he is an Apostate and Whoever doubts their kufr is himself a Kafir......[Sharh usul itiqaad Ahl ul sunnah 8/459] So Mr chacolate Yaseen Malik falls among them[Apotates] and rest jamaats too whom r supporting these kufrs wa murtadeens.","The Ijma a of the scholars upon a Disbelieving Ruler  Ruling Muslim. !!!The scholars all agree that the Imaamah (  rule ) of a disbeliever is not sound from outset. A disbeliever is not to have the reins of control over the affairs of the Muslims. Ibn al Mundhir (r.a) stated   Everyone I have memorized from of the people of knowledge agrees that a disbeliever is not to have authority over a Muslim under any circumstances.Takhrij : Ibn al Mundhir.Ibn Qayyim also stated the same in his book called.Now to the statement !! I came across one Talib ilm back two years ago. During the campaign and election of the country of Ghana. Said to me in a vocal statement. Oh we the Muslim need to vote in the Democratic elections to get rid off the kuffar government and elect a Muslim government. So I ask why he has that ideological  his statement was so that we can have a Muslim ruler. So my questions was which Hukum will be presented in the country? His answer was  it will take time to implement Shari ah because is Kuffar land. Automatically he has answer his own question which I m about to ask him.You want to used Kufr Akbar to eradicate kufr Akbar in Darul Kufr to establish Shari ah. Like Muhammed Mursi did ? Lol lol now where is he Now? In Fatah Sisi Prison  is Kufr and Shirk to vote for a kaafir to become your leader as a Muslim. Allah (swt) said in a Muhkamatu ayat يَا أَيُّهَا الَّذِينَ آمَنُوا أَطِيعُوا اللَّهَ وَأَطِيعُوا الرَّسُولَ وَأُولِي الْأَمْرِ مِنْكُمْ ۖ فَإِنْ تَنَازَعْتُمْ فِي شَيْءٍ فَرُدُّوهُ إِلَى اللَّهِ وَالرَّسُولِ إِنْ كُنْتُمْ تُؤْمِنُونَ بِاللَّهِ وَالْيَوْمِ الْآخِرِ ۚ ذَٰلِكَ خَيْرٌ وَأَحْسَنُ تَأْوِيلًا O you who believe! Obey Allah and obey the Messenger (Muhammad SAW)  and those of you (Muslims) who are in authority. (And) if you differ in anything amongst yourselves  refer it to Allah and His Messenger (SAW)  if you believe in Allah and in the Last Day. That is better and more suitable for final determination. Sura An Nisah   Ayah 59.When Allah (swt) used the word   AMONG YOU    MEANING   FROM YOU THE BELIEVERS.  WHOEVER S IS NOT A BELIEVER DOES NOT HAVE THE RIGHT OF OBEDIENCE. SALAFI","#Saudi Arabia is a land of Taghoot not Tawheed It s leader are people who have left the shariah of Allah n taken kuffar as its allies n other the. Shariah as its rule They have made Kufr their best friends n allies while they have made #Tawheed n Muslims as their enemies N u madcowlees can scream day n night how much u love ur King n Ahul salul but those of knowledge those who r not misguided by the deviant dog Rabi al #madkhalie know wat u r n knw that u r all murijah U hate Anwar al Awlaki Rahimhuallah n hate scholars who spread Tawheed and love Obama n Camron be aside they sleep with ur King n assist him n u","NEWS: MY SON has died but if all people stop sending their children to the Army  then who will fight for the nation?   Mohammad Haneef  father of the kidnapped and killed Kashmiri(Indian) army man. Death has to come one day. I had got him recruited in the Army to serve the nation. A soldier s job is to kill the enemy or get killed    he says. Murtadeen are more proud of their deaths for kufr than some Muslims are of their Shuhadah for Islam.","When you engage in kufr and human waste  the outcome is mainly two: 1. The rise or rather the more free environment for atheists and secularists. This includes: More freedom to publicly insult Allah and His Messengers and to bring out the alternative Islamic narrative that Pentagon is pushing for in a public and open way. Will attack and oppress Muslims who do not support Democratic systems and will mock Sunnah and it s people on a daily basis. Freedom will be limitless  so insults to Allah and His Messengers and His Deen becomes a fundamental human right. 2. The rise of misguidance like Sufism. This also includes other modern misguidance as well as the alternative narrative a bit toned down and hidden or subtle way. Somewhat against insults at least theoretically. Both outcomes will NOT stop nudity  fornication  interest  music and other vulgarities in fact it will increase and rules will help it grow under various names like youth  women  culture  art  etc.. This is what a rigged system looks like and those who engage in it are getting into these human waste whether one intended to do so or not","The majority of the people of kashmir are blind  blind  and again blind. They don t know what is kufr bit taghoot and Nullifiers of Islam  Importance of Al wala wal baraa. They talk based of emotions. They think based on emotions. They support based on emotions. They oppose based on emotions. They call apostates/ muslims.","Shiekh Ahmed ibn Umar Al Hazimi (may Allah hasten his release)  Kufr dune Kufr  Please share these videos ya ikhwan wa akhwat to the masses for dawah purposes the ummah needs it in sha Allah. Jazakamullahu khayran","The majority of the people of kashmir are blind  blind  and again blind. They don t know what is kufr bit taghoot and Nullifiers of Islam  Importance of Al wala wal baraa. They talk based of emotions. They think based on emotions. They support based on emotions. They oppose based on emotions. They call apostates/ muslims. Deviants / guided. Militants/ mujahideen. Rafidah s / Role model. Ismaili shia s / Qaid e Azam.","Nationalism is Kufr and those who ascribes themselves to be Nationalist are Murtadeen. Since it is narrated by Abu Dawud that the Messenger of Allah ﷺ said   He is not one of us who calls for `Asabiyah  (nationalism/tribalism) or fights for `Asabiyah or dies for [Sunan Abu Dawud (Vol. 2  pg. 753) No. 5121]`Asabiyah.","Christians and Shias have a lot in common  no surprise really  since kufr is one ugly Millah. (1)","One footed Mujahid in Mosul fight Millal Kufr  while Aaidh al Qarni wishes Shahadah for the sake of Allah in his bed https://t.co/lgWp0rRto1","@mykashmirmylife this is what our religion of allah says  . the one who supports kufr and shirk even with words are l… https://t.co/0OtM04lGQS"]}

datajson = json.dumps(data_terms)
response = requests.post("{}/terms".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

#___________________________________________________________________________________________________
print(Fore.BLUE+"Test /sento")
print(Style.RESET_ALL)

data = {'text': 'Allahum decrease the #fitnah between #Muslims and give patience to our #Muslim #lions who are fighting for the Deen', 'lang':'en','classifier':'Jihadist-English-Insikt.model','user_id':'berta01','case_id':'0053'}
datajson = json.dumps(data)
response = requests.post("{}/sento".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

data = {'text': 'los infieles deben morir en nombre de Ala para recuperar el Califato los leones lucharemos en la guerra y mataremos a todos los occidentales','classifier':'Jihadist-Spanish-Insikt.model'}
datajson = json.dumps(data)
response = requests.post("{}/sento".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

data = {'text':"les infidèles doivent mourir au nom d'Ala pour récupérer le califat, les lions vont se battre et tuer tous les occidentaux",'lang':'fr','classifier':'all'}
datajson = json.dumps(data)
response = requests.post("{}/sento".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

data = {'text': '#يوم_الجمعه ماهي_افضل_لعبه_عندك باقي_علي_الراتب هلـاكـ أكثر*من40مرتداً^من الـpkk*بهجمـآت مختلفة^لجنـ،ـود الخلـافة حكومة السامية', 'classifier':'Jihadist-Arabic-Insikt.model'}
datajson = json.dumps(data)
response = requests.post("{}/sento".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

data = {'text':'infidelii trebuie să moară în numele Ala pentru a recupera califatul, leii vor lupta în război și vor ucide toți occidentalii','lang':'ro','classifier':'Jihadist-Romanian-Insikt.model'}
datajson = json.dumps(data)
response = requests.post("{}/sento".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())


#___________________________________________________________________________________________________
print(Fore.BLUE+"Test /classifier")
print(Style.RESET_ALL)
classifier_data = {'annotated_data':[("Cats are very good hunters and use their strong, sharp claws and teeth to grab and hold their prey. In the wild, cats feed on mice, birds and other small animals.Senses","no"),("Cats see and hear extremely well","no"),("Cats can see in the dark and hear many sounds that humans are not able to hear","no"),("To feel their way round, cats use their whiskers","no"),("Cats say 'meow'","no"),("If cats feel very comfortable, they purr","no"),("If a cat is angry, it wags its tail, lowers its ears, and hisses or growls.","no"),("Just watch your cat: eyes, ears, tail and body posture tell you a lot about how your cat feels.","no"),("Things your cat needs:food and water - There are many different sorts of food. Ask your vet or pet shop for assistance. Make sure that your cat always has a bowl of fresh water.","no"),("Cats never drink where they eat, so put the water bowl at least one metre away from the food bowl.","no"),("a litter box and litter - Keep the litter box clean or you will soon have some stinky corners in your living room; a carrier - You'll have to see the vet from time to time. If you have to transport a cat, always use a carrier.","no"),("This can be a fancy cat's bed, the carrier or just an old cushion or blanket.","no"),("toys - Pet shops have lots of toys for cats, but even rolled-up wads of paper make nice toys. The most important thing is that you take your time to play with your cat.","no"),("a scratching post - A cat needs to scratch its claws and if you don't have a post, your furniture will suffer.","no"),("Our two cats are Ramses and Tari. Ramses is a tabby tom-cat. He is very relaxed and his favourite activities are eating and sleeping.","no"),("Tari is a black cat from the animal shelter. She is very active and loves playing and being snuggled. She doesn't like visitors, however - when someone comes to see us, she always hides away. While we are working on ego4u, Ramses and Tari are sleeping on the desk in front of the monitor.","no"),("Dogs (Canis lupus familiaris) are domesticated mammals, not natural wild animals.","yes"),("Dogs were originally bred from wolves. They have been bred by humans for a long time, and were the first animals ever to be domesticated.","yes"),("The dingo is also a dog, but many dingos have become wild animals again and live independently of humans in the range where they occur (parts of Australia).","yes"),("Today, some dogs are used as pets, others are used to help humans do their work.","yes"),("Dogs are a popular pet because they are usually playful, friendly, loyal and listen to humans.","yes"),("Thirty million dogs in the United States are registered as pets.","yes"),("Dogs eat both meat and vegetables, often mixed together and sold in stores as dog food.","yes"),("Dogs often have jobs, including as police dogs, army dogs, assistance dogs, fire dogs, messenger dogs, hunting dogs, herding dogs, or rescue dogs.","yes"),("They are sometimes called canines from the Latin word for dog - canis. Sometimes people also use dog to describe other canids, such as wolves.","yes"),("A baby dog is called a pup or puppy. A dog is called a puppy until it is about one year old.","yes"),("Dogs are sometimes referred to as man's best friend because they are kept as domestic pets and are usually loyal and like being around humans.","yes"),("Dogs like to be petted, but only when they can first see the petter's hand before petting; one should never pet a dog from behind.","yes"),("Dogs have four legs and make a bark,woof, or arf sound. Dogs often chase cats, and most dogs will fetch a ball or stick.","yes"),("Dogs can smell and hear better than humans, but cannot see well in color because they are color blind. Due to the anatomy of the eye, dogs can see better in dim light than humans. They also have a wider field of vision.","yes"),("Like wolves, wild dogs travel in groups called packs. Packs of dogs are ordered by rank, and dogs with low rank will submit to other dogs with higher rank. The highest ranked dog is called the alpha male. A dog in a group helps and cares for others. Domesticated dogs often view their owner as the alpha male.","yes"),("Different dog breeds have different lifespans. In general, smaller dogs live longer than bigger ones. The size and the breed of the dog change how long the dog lives, on average. Breeds such as the Dachshund usually live for fifteen years, Chihuahuas can reach age twenty. The Great Dane, on the other hand has an average lifespan of six to eight years; some Great Danes have lived for ten years.","yes"),("All dogs are descended from wolves, by domestication and artificial selection.","yes")] ,'lang':'en','user_id':'berta01','case_id':'0053','clas_name':'dogs'}

datajson = json.dumps(classifier_data)
response = requests.post("{}/classifier".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())

#___________________________________________________________________________________________________
print(Fore.BLUE+"Test /classifier - Not enought data")
print(Style.RESET_ALL)
classifier_data = {"annotated_data": [["Cats are very good hunters and use their strong, sharp claws and teeth to grab and hold their prey. In the wild, cats feed on mice, birds and other small #animals.Senses","no"],["All dogs are descended from wolves, by domestication and artificial selection.","yes"]],"lang":"en","user_id":"bubu","case_id":"5","clas_name":"dogs"}
datajson = json.dumps(classifier_data)
response = requests.post("{}/classifier".format(BASE_URL), data = datajson,  headers= header)
print(response.status_code)
print(response.json())


