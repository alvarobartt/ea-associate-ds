# Electronic Arts (EA) Assignment for: Associate Data Scientist

<img src="https://media-exp1.licdn.com/dms/image/C561BAQFjp6F5hjzDhg/company-background_10000/0?e=2159024400&v=beta&t=OfpXJFCHCqdhcTu7Ud-lediwihm0cANad1Kc_8JcMpA">

## Introduction

## Roadmap

## Content

    .
    ├── documents_challenge/          # Dataset of Multilingual Multi-Context documents
    ├── research/                     # Jupyter Notebooks and Reports of the project's research
    ├── slides/                       # Jupyter Slides for presenting the project 
    ├── 202007TestADS.pdf             # Electronia Arts (EA) Associate Data Scientist Assignment PDF File
    ├── LICENSE                       # MIT License so as to release the code open-source
    ├── README.md                     # Detailed README.md so as to explain the project
    └── requirements.txt              # Requirements to reproduce the Jupyter Notebooks

A description of the dataset and its building are described in the following paper:

[_A Multilingual, Multi-Style and Multi-Granularity Dataset for Cross-Language Textual Similarity Detection. Jérémy Ferrero, Frédéric Agnès, Laurent Besacier and Didier Schwab. In the 10th edition of the Language Resources and Evaluation Conference (LREC 2016)_](https://www.researchgate.net/publication/301861882_A_Multilingual_Multi-Style_and_Multi-Granularity_Dataset_for_Cross-Language_Textual_Similarity_Detection)

## Exploratory Data Analysis

## Text Preprocessing

```python
class CustomPreProcessor(object):
    """
    Custom PreProcessor

    Preprocesses the introduced raw text to transform it into clean text. This
    preprocessing pipe is regex based.

        >>> from apinlp.nlp.preprocessing import CustomPreProcessor
        >>> preprocessor = CustomPreProcessor()
        >>> print(preprocessor._preprocess("Visit us at https://www.ea.com/"))
        "visit us"
    """
    
    def __init__(self, strip_accents=True):
        self.strip_accents = strip_accents
        
        self.patterns = BASE_PATTERNS
        self.additional_patterns = (SPACES_PATTERN,)

        self.stopwords = STOPWORDS

    def _preprocess(self, text):
        """Cleans and applies a preprocessing layer to raw text"""
        text = text.replace('\t', ' ').replace('\n', ' ')
        
        if self.strip_accents:
            text = unidecode(text)

        for pattern in self.patterns:
            text = pattern.sub(' ', text)

        text = text.strip().lower()
        text = text.replace("'", " ")
        
        text = [word for word in text.split(' ') if len(word) > 2]

        for word in self.stopwords:
            text = list(filter((word.lower()).__ne__, text))

        text = ' '.join(text)
            
        for pattern in self.additional_patterns:
            text = pattern.sub(' ', text)
    
        return text
```

!["PreProcessed WordCloud"]("research/resources/preprocessed_wordcloud.png")

## Text Classification

!["Text Classification Models"]("research/resources/text_classification_models.png")

## Topic Modelling

!["Topic Modelling"]("research/resources/topic_modelling.png")

## Conclusions & Future Work

## References

