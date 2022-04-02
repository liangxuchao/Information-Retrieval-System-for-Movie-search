from unicodedata import category
from django import forms
Timechoice =(
    ("0", "Movie Length"),
    ("1", "Under 1 hour"),
    ("2", "1 ~ 2 hours"),
    ("3", "2 ~ 3 hours"),
    ("4", "More than 3 hours"),
)

Categorychoice =(
    ("0", "Select Category"),
)
Yearchoice=[("0", "Select Year"),
('2021','2021'),
         ('2020','2020'),
         ('2019','2019'),
         ('2018','2018'),
         ('2017','2017'),
         ('2016','2016'),
         ('2015','2015'),
         ('2014','2014')]

ShowRecordchoice=(
    ("20","20"),
    ("50","50"),
    ("100","100"),
    ("200","200"),
    ("all","ALL")
)


sortbychoice=(
    ("0","-- Select --"),
    ("Rate","Rate"),
    ("Votes","Votes"),
    ("Time", "Movie Length")
)
sortseqchoice=(
    ("desc","desc"),
    ("asc","asc"),
)

class HorizontalRadioSelect(forms.RadioSelect):
    template_name = 'horizontal_select.html'

class FreeTextForm(forms.Form):
    freetext = forms.CharField(required=False,widget=forms.TextInput(
        attrs={
            'list':"datalistOptions",
            'id':'freesearch',
        'class':'form-control',
        'placeholder':'Search...'
        }
    ))

    Movielength = forms.ChoiceField(widget=forms.Select(attrs={'class':'form-select','id':'timeselect'}),
    choices = Timechoice)
    
    # Category = forms.ChoiceField(widget=forms.Select(attrs={'class':'form-select','id':'categoryselect'}),
    # choices = Categorychoice)

    Year = forms.ChoiceField(widget=forms.Select(attrs={'class':'form-select','id':'Yearselect'}),
    choices = Yearchoice)
    
    Category = forms.MultipleChoiceField(required=False,widget=forms.CheckboxSelectMultiple())

    ShowRecords = forms.ChoiceField(widget=forms.Select(attrs={'class':'form-select','id':'ShowRecordselect'}),
    choices = ShowRecordchoice)

    sortby = forms.ChoiceField(widget=forms.Select(attrs={'class':'form-select','id':'sortbyselect'}),
    choices = sortbychoice)

    sortseq = forms.ChoiceField(widget=forms.Select(attrs={'class':'form-select','id':'sortseqselect'}),
    choices = sortseqchoice)

