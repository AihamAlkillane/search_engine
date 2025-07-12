document.addEventListener("DOMContentLoaded", () => {
  const modelSelect = document.getElementById("model-select");
  const searchInput = document.getElementById("search-input");
  const resultsContainer = document.getElementById("results-container");

  // TODO:: CHANGE models
  const aiModels = ["GPT-3", "GPT-4", "Gemini", "Claude", "Llama"];

  aiModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    modelSelect.appendChild(option);
  });

  const debounce = (func, delay) => {
    let timeoutId;
    return (...args) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        func.apply(this, args);
      }, delay);
    };
  };

  const search = async () => {
    const query = searchInput.value.trim();
    const selectedModel = modelSelect.value;

    if (query.length < 2) {
      resultsContainer.innerHTML =
        "<p>Please enter at least 2 characters to search.</p>";
      return;
    }

    resultsContainer.innerHTML = '<div class="loader">Loading...</div>';

    try {
      // TODO:: this is the API  
      const response = await fetch(
        `https://jsonplaceholder.typicode.com/comments?q=${query}`
      );
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      const data = await response.json();

      displayResults(data);
    } catch (error) {
      resultsContainer.innerHTML = `<p>Error fetching data: ${error.message}</p>`;
      console.error("Fetch error:", error);
    }
  };

  const displayResults = (data) => {
    resultsContainer.innerHTML = "";
    if (data.length === 0) {
      resultsContainer.innerHTML = "<p>No results found.</p>";
      return;
    }

    data.forEach((item) => {
      const resultItem = document.createElement("div");
      resultItem.classList.add("result-item");

      const title = document.createElement("h3");
      title.textContent = item.name;

      const body = document.createElement("p");
      body.textContent = item.body;

      const email = document.createElement("p");
      email.textContent = `By: ${item.email}`;
      email.style.fontSize = "12px";
      email.style.fontStyle = "italic";
      email.style.marginTop = "5px";

      resultItem.appendChild(title);
      resultItem.appendChild(body);
      resultItem.appendChild(email);
      resultsContainer.appendChild(resultItem);
    });
  };

  searchInput.addEventListener("input", debounce(search, 500));
}); 