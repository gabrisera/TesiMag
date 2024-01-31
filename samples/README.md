# Samples
Here you can find, for each data file, the samples create over them, the heuristic used for creating the sample is the following:
1. Creating the sliding windows over the data, each sliding windows have 103 values because we have for each feature (global_ke, head_ke, point_density,
   r_a_ke, l_a_ke, r_w_ke, l_w_ke, in totale 7 features) we have 14 features (according to the papers in this git) create over each feature such as :
     1. media  
     2. varianza
     3. MAD median absolute deviation
     4. max 
     5. min 
     6. sma signal magnitude area
     7. Energy (Average sum of squares)
     8. Iqr (interquartile range)
     9. signal Entropy
     10. coeff. corr (x1,x2)
     11. kurtosis signal (morbidezza rispetto a gaussiana del segnale (negativo piu morbido, positivo con piu picchi e code piu veloci))
     12. skewness (simmetria della distribuzione rispetto alla media)
     13. 14. kur ,skw in frequenza.
     (for point density and distal parts we habe also the repetitiveness)
   In total we have (7*14+5=103 features).
2. By using the heuristic suggested by Prof. Oneto i hade considered the samples over the saliencies and labled them at 1, and in the middle and labled at 0.
    
