document.getElementById('loadFileBtn').addEventListener('click', () => {
  const fileKey = document.getElementById("file_key").value.trim();

  if (!fileKey) {
    alert("Please enter a file key to load.");
    return;
  }

  fetch(`/load-file?file=${fileKey}`)
    .then(response => {
      if (!response.ok) {
        throw new Error("File could not be loaded.");
      }
      return response.text();
    })
    .then(data => {
      document.getElementById("filePreview").textContent = data;
    })
    .catch(error => {
      document.getElementById("filePreview").textContent = "Error loading file.";
    });
});
