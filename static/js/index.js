async function fetchData() {
    
    return [
        "Welcome to our website! Explore the wonders of AI-generated content with us.",
        "Curious about whether that captivating image was created by AI?",
        "Our advanced detection system will reveal the origins of the media,",
        "providing insights into the fascinating world of artificial intelligence creativity."
    ];
}


async function typewriterEffect(text, container) {
    const textElement = document.createElement('p');
    textElement.classList.add('typewriter');
    container.appendChild(textElement);
    
    for (let i = 0; i < text.length; i++) {
        textElement.textContent += text[i];
        await new Promise(resolve => setTimeout(resolve, 20)); 
    }
}

async function displayText() {
    const container = document.getElementById('typewriter-container');
    const texts = await fetchData();

    for (let text of texts) {
        await typewriterEffect(text, container);
        await new Promise(resolve => setTimeout(resolve, 100)); 
    }

    
    const buttonContainer = document.getElementById('button-container');
    buttonContainer.style.display = 'block';
    const button = document.createElement('button');
    button.classList.add('btn', 'fade-in');
    button.textContent = 'Get Started';
    button.addEventListener('click', () => {
        console.log('Button clicked! Add your functionality here.');
    });
    buttonContainer.appendChild(button);
}

displayText();
document.addEventListener('DOMContentLoaded', () => {
    const navLinks = document.querySelectorAll('.main-nav a');

    window.addEventListener('scroll', () => {
        let current = '';

        navLinks.forEach(link => {
            const section = document.querySelector(link.getAttribute('href'));
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;

            if (window.scrollY >= sectionTop - sectionHeight / 3) {
                current = link.getAttribute('href');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === current) {
                link.classList.add('active');
            }
        });
    });
});

