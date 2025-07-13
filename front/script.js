document.addEventListener("DOMContentLoaded", () => {
  const modelSelect = document.getElementById("model-select");
  const modelTypeSelect = document.getElementById("model-type-index");
  const searchInput = document.getElementById("search-input");
  const resultsContainer = document.getElementById("results-container");

  const aiModels = ["tf-idf", "bert-sentence", "hybrid-serial", "hybrid-parallel"];
  const modelTypeIndexes = ["inverted_index", "flat_ip_index"];

  aiModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    modelSelect.appendChild(option);
  });

  modelTypeIndexes.forEach((modelTypeIndex) => {
    const option = document.createElement("option");
    option.value = modelTypeIndex;
    option.textContent = modelTypeIndex;
    modelTypeSelect.appendChild(option);
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
    const model_type = modelSelect.value;
    const index_type = modelTypeSelect.value;

    if (query.length < 2) {
      resultsContainer.innerHTML = "<p>Please enter at least 2 characters to search.</p>";
      return;
    }

    resultsContainer.innerHTML = '<div class="loader">Loading...</div>';

    try {
      const response = await fetch("http://localhost:8080/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query, model_type, index_type }),
      });

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
    if (!data || data.length === 0) {
      resultsContainer.innerHTML = "<p>No results found.</p>";
      return;
    }

    data.forEach((item) => {
      const resultItem = document.createElement("div");
      resultItem.classList.add("result-item");

      const title = document.createElement("h3");
      title.textContent = item.title || item.name || "No title";

      const body = document.createElement("p");
      body.textContent = item.body || item.text || "";

      resultItem.appendChild(title);
      resultItem.appendChild(body);
      resultsContainer.appendChild(resultItem);
    });
  };

  searchInput.addEventListener("input", debounce(search, 500));
});
