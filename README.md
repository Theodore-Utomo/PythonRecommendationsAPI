Example use in JavaScript.
Make sure `restaurantName` is equivalent to one of the restaurants in the database.

```javascript
async function getRestaurantRecommendations(restaurantName) {
  const response = await fetch(
    `https://pythonrecommendationsapi.onrender.com//recommend?name=${encodeURIComponent(
      restaurantName
    )}`
  );
  if (!response.ok) {
    console.error("Error fetching recommendations:", response.statusText);
    return;
  }
  const recommendations = await response.json();
  console.log("Recommendations:", recommendations);
  return recommendations;
}
```

Returns an array of json restaurant objects.

