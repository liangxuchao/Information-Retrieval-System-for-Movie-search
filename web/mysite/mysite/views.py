from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.template import loader

from mysite.forms import FreeTextForm
import requests
from mysite.genreClassification.genreClassifyBytfidf import genreClassifyBytfidf
from mysite.genreClassification.genreClassifyByDLbidirdirection import genreClassifyByDLbidirdirection
import os
import csv
from pathlib import Path
import pandas as pd
baseApiUrl = "http://localhost:8983/solr/CSVCore"

def pre_processign_query(l):
    
    remove_characters = ["-", "/", ",",":","@","!"]

    for character in remove_characters:
        l = l.replace(character, " ")

    listword = l.split()

    stopwords = ["the","a","an","and","are","as","at","be","but","by","for","if","in","into","is","it","no","not","of","on","or","such","that","the","their","then","there","these","they","this","to","was","will","with"]
    removestopwords  = [word for word in listword if word.lower() not in stopwords]

    ulist = []
    [ulist.append(x) for x in removestopwords if x not in ulist]

    print(ulist)
    return ulist
    

def index(request):
    getCategory =  baseApiUrl + "/select?q=*:*&facet=on&facet.field=Genre&rows=0"
    print(getCategory)
    r2 = requests.get(getCategory, params=request.GET).json()
    Genre = r2["facet_counts"]["facet_fields"]["Genre"]
    
    choice = []
    for i in range(0,len(Genre) -1,2):
        choice.append((Genre[i],Genre[i] + " (" + str(Genre[i+1]) + ")"))
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = FreeTextForm(request.POST)
        # check whether it's valid:
        form.fields["Category"].choices=choice

        if form.is_valid():
            
            searchValue = form.cleaned_data['freetext']
            Year = form.cleaned_data['Year']
            Category = form.cleaned_data['Category']
            MovieLength = form.cleaned_data['Movielength']
            ShowRecords = form.cleaned_data['ShowRecords']

            sortseq = form.cleaned_data['sortseq']
            sortby = form.cleaned_data['sortby']

            print(Category)
            defaultparameter = "q.op=OR&q=*:*"
            freetextSearchParameter = ""
            otherparameter = ""
            print(otherparameter)
            searchValue=' '.join(pre_processign_query(searchValue))
            suggeststringShow = searchValue

            if searchValue == "" and Year == "0" and Category == len(Category) and MovieLength == "0":
                
                otherparameter = otherparameter + "&fq=Rate:[8 TO *]&fq=Votes:[100000 TO *]"
                
            else:
                fsearchValue = searchValue
                if searchValue != "":
                    if ' ' in searchValue:
                        fsearchValue = '"' + searchValue + '"' + "~10"
                    else:
                        fsearchValue = '*' + searchValue + '*'

                    
                    freetextSearchParameter = freetextSearchParameter + "&fq=term:" + fsearchValue  
                
                if Year != "0":
                    otherparameter = otherparameter + "&fq=Year:" + Year
                
                if len(Category) != 0:
                    urlstr = "("
                    
                    for i in range(0,len(Category) - 1):
                        urlstr = urlstr + Category[i] + " "
                    urlstr = urlstr + Category[len(Category) - 1] + ")"
                    otherparameter = otherparameter + '&fq=Genre:' + urlstr

                if MovieLength != "0":
                    timestart = "0"
                    timeend = "0"
                    if MovieLength == "1":
                        timeend = "59"
                    elif MovieLength == "2":
                        timestart = "60"
                        timeend = "119"
                    elif MovieLength == "3":
                        timestart = "120"
                        timeend = "179"
                    elif MovieLength == "4":
                        timestart = "180"
                        timeend = "*"
                    otherparameter = otherparameter + "&fq=Time:[" + timestart + " TO " + timeend + "]"

            if ShowRecords != "all":
                otherparameter = otherparameter + "&rows=" + ShowRecords
            else:
                otherparameter = otherparameter + "&rows=200000"

            if sortby != "0":
                otherparameter = otherparameter + "&sort=" + sortby + " " + sortseq
            else:
                otherparameter = otherparameter + "&sort=Votes desc,Rate desc,Year desc"
                

            finalUrl = baseApiUrl + "/select?" + defaultparameter + freetextSearchParameter + otherparameter
            #print(finalUrl)
            
            print(finalUrl)
            r = requests.get(finalUrl, params=request.GET).json()

            finalresponse = r
            suggeststring = ""
            #print(r)
            if searchValue != "" and r["response"]["numFound"] == 0 :
                searchValueArr = searchValue.split()
                for i in range(0,len(searchValueArr)):
                    searchword =searchValueArr[i]
                    spellcheckUrl = baseApiUrl + "/spell?q=" + searchword
                    print(spellcheckUrl)
                    s = requests.get(spellcheckUrl, params=request.GET).json()
                    
                    print(s)
                    if "spellcheck" in s and len(s["spellcheck"]["suggestions"]) >0:
                        highword = s["spellcheck"]["suggestions"][1]["suggestion"][0]['word']
                        highfreq = s["spellcheck"]["suggestions"][1]["suggestion"][0]['freq']
                        if len(s["spellcheck"]["suggestions"][1]["suggestion"]) > 1:
                            for x in range(0,len(s["spellcheck"]["suggestions"][1]["suggestion"])):
                                
                                if s["spellcheck"]["suggestions"][1]["suggestion"][x]['freq'] > highfreq:
                                    highfreq = s["spellcheck"]["suggestions"][1]["suggestion"][x]['freq']
                                    highword = s["spellcheck"]["suggestions"][1]["suggestion"][x]['word']
                        suggeststring = suggeststring + highword + " "
                    else: 
                        suggeststring = suggeststring + searchValueArr[i] + " "
                
                suggeststring = suggeststring.rstrip()
                newsearchstring =""
                if len(searchValueArr) > 1:
                        newsearchstring = '"' + suggeststring + '"' + "~10"
                else:
                        newsearchstring = '*' + suggeststring + '*'

                freetextSearchParameter = "&fq=term:" + newsearchstring  
                
                re_searchUrl = baseApiUrl + "/select?" + defaultparameter + freetextSearchParameter + otherparameter  + "&sort=Votes desc,Rate desc,Year desc" 
                
                print(re_searchUrl) 
                #print(finalUrl)
                re = requests.get(re_searchUrl, params=request.GET).json()
                print("\n")
                print(re)
                if re["response"]["numFound"] > 0:
                    suggeststringShow = suggeststring
                    finalresponse = re
            
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            print(suggeststringShow)
            return render(request, 'index.html', {'form': form, 'result':finalresponse["response"], 'suggeststringShow':suggeststringShow, 'originalsearchstring':form.cleaned_data['freetext']})

    # if a GET (or any other method) we'll create a blank form
    else:
        finalUrl = baseApiUrl + "/select?fq=Rate:[8 TO *]&fq=Votes:[100000 TO *]&q.op=OR&q=*:*&rows=20&sort=Votes desc, Rate desc, Year desc" 
        r = requests.get(finalUrl, params=request.GET).json()
        #print(finalUrl)
        form = FreeTextForm()
        
        form.fields["Category"].choices=choice
        return render(request, 'index.html', {'form': form, 'result':r["response"], 'suggeststringShow':"", 'originalsearchstring':""})

def genreclassifytfidf(request):
    
    path = str(Path(__file__).resolve().parent.parent) + "\data.csv"
    print(path)
    genreClass = genreClassifyBytfidf(path)
    genreClass.pre_processing()

    predictionArr, f1 =  genreClass.predict(100)
    # genreClass.predict(50)
    context={}
    context['genreSummaryGraph'] =  genreClass.print_genre_summary()
    context['freqwordSummaryGraph'] =  genreClass.print_freq_words()
    context['predictionArr'] =  predictionArr
    context['f1'] =  f1
    return render(request, 'genretfidf.html', context)


def genreclassifybd(request):
    hl, score, precision,recall,f1 = 0,0,0,0,0
    path = str(Path(__file__).resolve().parent) + r"\genreClassification\bdresult.csv"
    datapath = str(Path(__file__).resolve().parent.parent) + "\data.csv"  
    rows = []
    if request.method == 'POST':
        path = str(Path(__file__).resolve().parent.parent) + "\data.csv"
        print(path)
        train_df = pd.read_csv(datapath)
        genreClass = genreClassifyByDLbidirdirection()
        hl, score, precision,recall,f1 = genreClass.train_model(train_df)
        predict = genreClass.predict(train_df.sample(n=100))
        for row in predict:
            rows.append(row)
    else:
        file = open(path)
        csvreader = csv.reader(file)
        print(csvreader)
        try:
            next(csvreader)
        except StopIteration:
            has_headers = False
        
        for row in csvreader:
            rows.append(row)
        print(rows)
        file.close()
    context={}
    context["predictionArr"] =rows
    context["Hammingloss"] =hl
    context["Score"] =score
    context["Precision"] =precision
    context["Recall"] =recall
    context["F1"] =f1
    return render(request, 'genrebd.html', context)