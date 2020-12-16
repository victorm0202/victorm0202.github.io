#### <img src="figs/logocimat.png" height="20%" width="20%"  align="center"/>

# <center> Métodos de aprendizaje automático para análisis de textos<center>

<center> Víctor Muñiz Sánchez<center>
<center> Diciembre 2020<center>

# Sobre el curso

## Objetivos:

* Mostrar los conceptos básicos de Procesamiento de Lenguaje Natural (NLP) orientado a textos.
* Mostrar repersentaciones vectoriales útiles de textos a partir de modelos probabilísticos de lenguaje y modelos neuronales de lenguaje.
* Abordar modelos de aprendizaje supervisado y no supervisado utilizando métodos de Machine Learning y Deep Learning, dando especial énfasis en arquitecturas de aprendizaje profundo y diversas aplicaciones.
    
    

# Temario

2. Conceptos básicos de python para el curso
3. Introducción y conceptos básicos de NLP
4. Representaciones básicos de textos: one-hot encoding, modelo n-gram, bolsa de palabras y TF-IDF.
5. Embeddings para palabras y documentos basados en modelos neuronales de lenguaje
6. Modelos de ML y aplicaciones en textos
7. Modelos de DL para textos: redes convolucionales, recurrentes y aplicaciones
8. Arquitecturas avanzadas de DL: sequence to sequence y mecanismos de atención (si da tiempo...)

# Introducción

NLP (Jurafsky & Martin, Speech and Language Processing, 2nd. Ed.
Es un campo de estudio enfocado en la interacción entre __lenguaje humano__ y computadoras. Se encuentra en la intersecciónn de ciencias de la computación, inteligencia artificial y linguistica computacional.
    
El objetivo es que las computadoras, realicen tareas útiles que involucren lenguaje humano, como comunicación máquina-humano, mejorar la comunicación humano-humano o simplemente, realizar procesamiento útil de texto o discurso.

Concepto clave: __lenguaje humano__:
\begin{itemize}
\item Signos linguísticos
\item Signos gráficos (textual)
\item Secuencias sonoras
\item Gestos y señales
\end{itemize}

Nosotros hablaremos sobre textos
<img src="figs/noest3.png" height="35%" width="35%" align="center"/>

__NLP es un área bastante compleja__. Esto se debe principalmente, a que el lenguaje natural es complejo en sí:

\begin{itemize}
\item Altamente ambiguo
\item Utiliza procesos mentales complejos para obtener un significado (uso del entorno)
\item Considera diferentes tipos de "entradas": texto, audio, imágenes, expresiones faciales y corporales, otras representaciones pictóricas 😮 👏 🙌 
\item Resultados en tiempo real (machine translation, automatic answering, etc...).
\item En constante evolución
\end{itemize}

También es un área bastante amplia.

¿Cuántas tareas/aplicaciones de NLP para textos conoces?

Veamos lo que dice Wikipedia


```python
import wikipedia
wikipedia.set_lang("en")
print(wikipedia.search("natural language processing"))
```

    ['Natural language processing', 'History of natural language processing', 'Natural language', 'Natural-language understanding', 'Outline of natural language processing', 'Natural-language user interface', 'Natural Language Toolkit', 'Process', 'List of artificial intelligence projects', 'GPT-3']

