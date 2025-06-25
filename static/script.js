const form = document.getElementById('registerForm');
form.addEventListener('submit', async function(e) {
  e.preventDefault();
  const formData = new FormData(form);
  const responseDiv = document.getElementById('response');

  try {
    const res = await fetch('/register', {
      method: 'POST',
      body: formData
    });

    const result = await res.json();

    if (res.ok) {
      responseDiv.innerHTML = `✅ ${result.message}<br><br><em>${result.welcome}</em>`;
      responseDiv.style.color = 'green';
    } else {
      responseDiv.innerText = result.error || 'Something went wrong.';
      responseDiv.style.color = 'red';
    }
  } catch (err) {
    responseDiv.innerText = '⚠️ Failed to connect to server.';
    responseDiv.style.color = 'red';
  }
});
