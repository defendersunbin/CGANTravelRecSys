CGAN Travel Recommendation System

This system is travel recommendation system made with Conditional Generative Adversarial Network (CGAN).
Main contribution of this system is to solve the cold-start challenge of sparse data travelers (i.e., traveler with no travel history, only travel for short time with places.)

This system has a novel algorithm made by defendersunbin, an 'Anchor-and-Expand' algorithm for cold-start user who have 'zero' travel history. Yes, zero (none). You can check out my dataset and scroll all the way down and find userID: 111111. That is the cold-start user with no travel history.
This algorithm works by: 
1. Gather all information from a traveler with histories including visit frequency, GPS coordinates, theme/subtheme of POI places, users' preference of the POI theme/subtheme, sequence of patterns from all users.
2. Train with CGAN. CGAN analyze the information from the user and made a concept what kind of preference does this user likes, and look through the sequence of POI with route pattern and simultaneously finds out where the user hasn't visited + where the other similar users went.
3. Then it searches for anchor POI as a start point. And the candidate of the start point is all POIs inside the datasets.
4. After it searches for anchor POI, it starts to expand the route with the specific pattern where that user been before and recommend to the novel user. Duplicate POI and routes are not generated.
5. Finally, it show the routes with POIs and show it from a map. You can check out the generated route for the cold-start user from cold-start generated routes folder. I've tested out with 5 cities (Edinbourgh, Glasgow, Melbourne, Toronto, Osaka) and it works like a charm.

You can find the results by clicking 5 cities folder (Edinburgh, Melbourne, Glasgow, Toronto, Osaka).
This includes the training results of CGAN, t-SNE, dimensions.
