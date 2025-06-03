document.addEventListener('DOMContentLoaded', function() {
    
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    
    

    
    initMovingParticlesBackground();
    initHeroParticles();
    initWebParticles();
    initScrollAnimations();
    initSmoothScrolling();
    initMobileMenu();
    initCounterAnimations();
    initCursorTrail();
    initStickyNavbar();
    initTestimonialCarousel();
    
    initFeatureTabs();
    initNotifications();
    
    
    checkFlashMessages();
    
    
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        setTimeout(() => {
            card.style.transition = 'all 0.6s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
});


function initMovingParticlesBackground() {
    const container = document.getElementById('particlesBackground');
    if (!container) return;
    
    
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        return;
    }
    
    const particleCount = window.innerWidth > 768 ? 40 : 20;
    
    for (let i = 0; i < particleCount; i++) {
        createBackgroundParticle(container);
    }
    
    
    setInterval(() => {
        if (container.children.length < particleCount) {
            createBackgroundParticle(container);
        }
    }, 3000);
}

function createBackgroundParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    
    
    particle.style.left = Math.random() * 100 + '%';
    
    
    const size = Math.random() * 4 + 2;
    particle.style.width = size + 'px';
    particle.style.height = size + 'px';
    
    
    particle.style.animationDelay = Math.random() * 5 + 's';
    particle.style.animationDuration = (Math.random() * 10 + 15) + 's';
    
    
    particle.style.opacity = Math.random() * 0.6 + 0.2;
    
    container.appendChild(particle);
    
    
    const animationDuration = parseFloat(particle.style.animationDuration) * 1000;
    setTimeout(() => {
        if (particle.parentNode) {
            particle.parentNode.removeChild(particle);
        }
    }, animationDuration);
}


function initHeroParticles() {
    const container = document.getElementById('heroParticles');
    if (!container) return;
    
    
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        return;
    }
    
    const particleCount = 15;
    
    for (let i = 0; i < particleCount; i++) {
        createHeroParticle(container);
    }
    
    
    setInterval(() => {
        if (container.children.length < particleCount) {
            createHeroParticle(container);
        }
    }, 2000);
}

function createHeroParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'hero-particle';
    
    
    particle.style.left = Math.random() * 100 + '%';
    
    
    const size = Math.random() * 5 + 3;
    particle.style.width = size + 'px';
    particle.style.height = size + 'px';
    
    
    particle.style.animationDelay = Math.random() * 3 + 's';
    particle.style.animationDuration = (Math.random() * 6 + 12) + 's';
    
    
    particle.style.opacity = Math.random() * 0.8 + 0.2;
    
    container.appendChild(particle);
    
    
    const animationDuration = parseFloat(particle.style.animationDuration) * 1000;
    setTimeout(() => {
        if (particle.parentNode) {
            particle.parentNode.removeChild(particle);
        }
    }, animationDuration);
}


function initWebParticles() {
    const container = document.getElementById('webParticles');
    if (!container) return;
    
    
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        return;
    }
    
    const particleCount = window.innerWidth > 768 ? 60 : 30;
    
    
    for (let i = 0; i < particleCount; i++) {
        createWebParticle(container);
    }
    
    
    for (let i = 0; i < 15; i++) {
        createWebLine(container);
    }
    
    
    setInterval(() => {
        if (container.querySelectorAll('.web-particle').length < particleCount) {
            createWebParticle(container);
        }
        if (container.querySelectorAll('.web-line').length < 15) {
            createWebLine(container);
        }
    }, 2000);
}

function createWebParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'web-particle';
    
    
    particle.style.left = Math.random() * 100 + '%';
    
    
    const size = Math.random() * 3 + 2;
    particle.style.width = size + 'px';
    particle.style.height = size + 'px';
    
    
    particle.style.animationDelay = Math.random() * 5 + 's';
    particle.style.animationDuration = (Math.random() * 10 + 15) + 's';
    
    
    const colors = [
        'rgba(139, 92, 246, 0.8)',
        'rgba(6, 182, 212, 0.6)',
        'rgba(236, 72, 153, 0.7)',
        'rgba(255, 255, 255, 0.5)'
    ];
    particle.style.background = colors[Math.floor(Math.random() * colors.length)];
    
    container.appendChild(particle);
    
    
    const animationDuration = parseFloat(particle.style.animationDuration) * 1000;
    setTimeout(() => {
        if (particle.parentNode) {
            particle.parentNode.removeChild(particle);
        }
    }, animationDuration);
}

function createWebLine(container) {
    const line = document.createElement('div');
    line.className = 'web-line';
    
    
    line.style.left = Math.random() * 80 + 10 + '%';
    line.style.top = Math.random() * 80 + 10 + '%';
    
    
    line.style.width = Math.random() * 100 + 50 + 'px';
    
    
    line.style.transform = `rotate(${Math.random() * 360}deg)`;
    
    
    line.style.animationDelay = Math.random() * 3 + 's';
    
    container.appendChild(line);
    
    
    setTimeout(() => {
        if (line.parentNode) {
            line.parentNode.removeChild(line);
        }
    }, 10000);
}


function initScrollAnimations() {
    const animatedElements = document.querySelectorAll('.animate-fade-in, .animate-slide-up, .animate-slide-left, .animate-slide-right, .animate-scale');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;
                const delay = element.getAttribute('data-delay') || 0;
                
                setTimeout(() => {
                    element.classList.add('in-view');
                }, parseInt(delay));
                
                
                observer.unobserve(element);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    animatedElements.forEach(element => {
        observer.observe(element);
    });
}

// Smooth Scrolling
function initSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href === '#') return;
            
            const target = document.querySelector(href);
            if (target) {
                e.preventDefault();
                
                const offsetTop = target.offsetTop - 100; 
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}


function initCounterAnimations() {
    let countersAnimated = false;
    
    const statNumbers = document.querySelectorAll('.stat-number');
    
    function animateCounters() {
        if (countersAnimated) return;
        
        statNumbers.forEach(stat => {
            const target = stat.textContent.replace(/\D/g, ''); 
            if (target) {
                animateCounter(stat, 0, parseInt(target), 2000);
            }
        });
        
        countersAnimated = true;
    }
    
    function animateCounter(element, start, end, duration) {
        const range = end - start;
        const increment = end > start ? 1 : -1;
        const stepTime = Math.abs(Math.floor(duration / range));
        
        let current = start;
        const timer = setInterval(() => {
            current += increment;
            element.textContent = current + (element.textContent.includes('+') ? '+' : '');
            
            if (current === end) {
                clearInterval(timer);
            }
        }, stepTime);
    }
    
    
    const statsSection = document.querySelector('.stats-section');
    if (statsSection) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    setTimeout(animateCounters, 500);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });
        
        observer.observe(statsSection);
    }
}


function initMobileMenu() {
    const mobileToggle = document.querySelector('.mobile-menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    
    if (mobileToggle && sidebar) {
        mobileToggle.addEventListener('click', function() {
            sidebar.classList.toggle('active');
            document.body.classList.toggle('sidebar-open');
        });
        
        
        document.addEventListener('click', function(e) {
            if (!sidebar.contains(e.target) && !mobileToggle.contains(e.target)) {
                sidebar.classList.remove('active');
                document.body.classList.remove('sidebar-open');
            }
        });
    }
}


function initCursorTrail() {
    if (window.innerWidth <= 768) return; 
    
    let mouseTrail = [];
    const trailLength = 10;
    
    document.addEventListener('mousemove', function(e) {
        mouseTrail.push({ x: e.clientX, y: e.clientY });
        
        if (mouseTrail.length > trailLength) {
            mouseTrail.shift();
        }
        
        updateTrail();
    });
    
    function updateTrail() {
        
        document.querySelectorAll('.cursor-trail').forEach(el => el.remove());
        
        mouseTrail.forEach((point, index) => {
            if (index % 2 === 0) { 
                createTrailPoint(point.x, point.y, index);
            }
        });
    }
    
    function createTrailPoint(x, y, index) {
        const trail = document.createElement('div');
        trail.className = 'cursor-trail';
        trail.style.cssText = `
            position: fixed;
            left: ${x}px;
            top: ${y}px;
            width: 4px;
            height: 4px;
            background: rgba(139, 92, 246, ${0.8 - (index * 0.08)});
            border-radius: 50%;
            pointer-events: none;
            z-index: 9999;
            transition: opacity 0.3s ease;
        `;
        
        document.body.appendChild(trail);
        
        setTimeout(() => {
            trail.style.opacity = '0';
            setTimeout(() => trail.remove(), 300);
        }, 100);
    }
}


function initStickyNavbar() {
    const navbar = document.querySelector('.top-navbar');
    if (!navbar) return;
    
    let lastScrollY = window.scrollY;
    
    window.addEventListener('scroll', throttle(() => {
        const currentScrollY = window.scrollY;
        
        if (currentScrollY > 100) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        
        lastScrollY = currentScrollY;
    }, 16));
}


function initTestimonialCarousel() {
    const carousel = document.querySelector('#testimonialsCarousel');
    if (carousel) {
        
        setInterval(() => {
            const nextButton = carousel.querySelector('.carousel-control-next');
            if (nextButton) {
                nextButton.click();
            }
        }, 8000);
    }
}


function initFAQAccordion() {
    
    
    
    
    
    const accordionButtons = document.querySelectorAll('.accordion-button');
    
    accordionButtons.forEach(button => {
        button.addEventListener('click', function() {
            
            this.classList.add('accordion-clicking');
            setTimeout(() => {
                this.classList.remove('accordion-clicking');
            }, 150);
        });
    });
}


function initFeatureTabs() {
    const tabButtons = document.querySelectorAll('.nav-tabs .nav-link');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            
            const targetPane = document.querySelector(this.getAttribute('data-bs-target'));
            if (targetPane) {
                targetPane.style.opacity = '0';
                setTimeout(() => {
                    targetPane.style.opacity = '1';
                }, 150);
            }
        });
    });
}


function showNotification(message, type = 'info', duration = 4000) {
    const notification = document.createElement('div');
    notification.className = `cosmic-notification cosmic-notification-${type}`;
    
    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-triangle',
        warning: 'fas fa-exclamation-circle',
        info: 'fas fa-info-circle'
    };
    
    notification.innerHTML = `
        <div class="notification-content">
            <i class="${icons[type]} me-2"></i>
            <span class="notification-message">${message}</span>
        </div>
        <button type="button" class="notification-close" onclick="closeNotification(this.parentElement)">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    
    let container = document.querySelector('.notification-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'notification-container';
        document.body.appendChild(container);
    }
    
    container.appendChild(notification);
    
    
    setTimeout(() => notification.classList.add('show'), 100);
    
    
    if (duration > 0) {
        setTimeout(() => closeNotification(notification), duration);
    }
    
    return notification;
}

function closeNotification(notification) {
    if (notification && notification.parentNode) {
        notification.classList.add('hiding');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }
}


function showToast(message, type = 'info', duration = 3000) {
    return showNotification(message, type, duration);
}

function getToastIcon(type) {
    const icons = {
        success: 'check-circle',
        error: 'exclamation-triangle',
        warning: 'exclamation-circle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

function closeToast(toast) {
    return closeNotification(toast);
}


function initNotifications() {
    
    const urlParams = new URLSearchParams(window.location.search);
    
    if (urlParams.get('logged_in') === 'true') {
        showToast('Successfully logged in!', 'success');
    }
    
    if (urlParams.get('logged_out') === 'true') {
        showToast('Successfully logged out!', 'info');
    }
    
    if (urlParams.get('registered') === 'true') {
        showToast('Account created successfully!', 'success');
    }
}


function checkFlashMessages() {
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(alert => {
        const text = alert.textContent.trim();
        const type = alert.classList.contains('alert-danger') ? 'error' : 
                    alert.classList.contains('alert-success') ? 'success' :
                    alert.classList.contains('alert-warning') ? 'warning' : 'info';
        
        if (text) {
            showNotification(text, type);
        }
        
        
        alert.style.display = 'none';
    });
    
    
    if (window.location.pathname === '/') {
        setTimeout(() => {
            showNotification('Welcome to GradeUP! Notification system is working.', 'info', 3000);
        }, 1000);
    }
}


function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

function debounce(func, wait, immediate) {
    let timeout;
    return function() {
        const context = this, args = arguments;
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
} 