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
    initMobileOptimizations();
    
    
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
    
    // Check for reduced motion preference
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        return;
    }
    
    // Reduce particle count on mobile for better performance
    let particleCount;
    if (window.innerWidth <= 480) {
        particleCount = 15; // Very small screens
    } else if (window.innerWidth <= 768) {
        particleCount = 25; // Mobile devices
    } else {
        particleCount = 40; // Desktop
    }
    
    for (let i = 0; i < particleCount; i++) {
        createBackgroundParticle(container);
    }
    
    // Adjust interval based on device capabilities
    const intervalTime = window.innerWidth <= 768 ? 4000 : 3000;
    setInterval(() => {
        if (container.children.length < particleCount) {
            createBackgroundParticle(container);
        }
    }, intervalTime);
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
    let overlay = document.querySelector('.mobile-overlay');
    
    if (!mobileToggle || !sidebar) return;
    
    // Create overlay if it doesn't exist
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'mobile-overlay';
        document.body.appendChild(overlay);
    }
    
    let previousFocus = null;
    
    // Toggle mobile menu
    mobileToggle.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        toggleMobileMenu();
    });
    
    // Close menu when clicking overlay
    overlay.addEventListener('click', function() {
        closeMobileMenu();
    });
    
    // Handle escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && sidebar.classList.contains('active')) {
            closeMobileMenu();
        }
    });
    
    // Close menu when navigation link is clicked on mobile
    const navLinks = sidebar.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            if (window.innerWidth <= 1024) {
                setTimeout(() => closeMobileMenu(), 100);
            }
        });
    });
    
    // Handle window resize
    window.addEventListener('resize', throttle(function() {
        if (window.innerWidth > 1024) {
            closeMobileMenu();
        }
    }, 250));
    
    // Add swipe gestures for mobile
    if ('ontouchstart' in window) {
        initSwipeGestures(sidebar);
    }
    
    function toggleMobileMenu() {
        const isOpen = sidebar.classList.contains('active');
        
        if (isOpen) {
            closeMobileMenu();
        } else {
            openMobileMenu();
        }
    }
    
    function openMobileMenu() {
        previousFocus = document.activeElement;
        
        sidebar.classList.add('active');
        overlay.classList.add('active');
        document.body.classList.add('sidebar-open');
        
        // Prevent body scroll on mobile
        if (window.innerWidth <= 1024) {
            document.body.style.overflow = 'hidden';
        }
        
        // Update ARIA attributes
        mobileToggle.setAttribute('aria-expanded', 'true');
        sidebar.setAttribute('aria-hidden', 'false');
        
        // Update menu icon
        updateMenuIcon(true);
        
        // Focus management
        setTimeout(() => {
            const firstFocusable = sidebar.querySelector('.nav-link, button');
            if (firstFocusable) {
                firstFocusable.focus();
            }
        }, 100);
    }
    
    function closeMobileMenu() {
        sidebar.classList.remove('active');
        overlay.classList.remove('active');
        document.body.classList.remove('sidebar-open');
        
        // Restore body scroll
        document.body.style.overflow = '';
        
        // Update ARIA attributes
        mobileToggle.setAttribute('aria-expanded', 'false');
        sidebar.setAttribute('aria-hidden', 'true');
        
        // Update menu icon
        updateMenuIcon(false);
        
        // Restore focus
        if (previousFocus) {
            previousFocus.focus();
            previousFocus = null;
        } else {
            mobileToggle.focus();
        }
    }
    
    function updateMenuIcon(isOpen) {
        const icon = mobileToggle.querySelector('i');
        if (icon) {
            icon.className = isOpen ? 'fas fa-times' : 'fas fa-bars';
        }
    }
    
    function initSwipeGestures(element) {
        let startX = 0;
        let startY = 0;
        let distX = 0;
        let distY = 0;
        let threshold = 120; // Reduced threshold for easier swiping
        let restraint = 100;
        let allowedTime = 300;
        let startTime = 0;
        
        // Swipe on sidebar
        element.addEventListener('touchstart', function(e) {
            const touchObj = e.changedTouches[0];
            startX = touchObj.pageX;
            startY = touchObj.pageY;
            startTime = new Date().getTime();
        }, {passive: true});
        
        element.addEventListener('touchmove', function(e) {
            // Prevent scrolling when swiping horizontally
            const touchObj = e.changedTouches[0];
            const currentDistX = Math.abs(touchObj.pageX - startX);
            const currentDistY = Math.abs(touchObj.pageY - startY);
            
            if (currentDistX > currentDistY && currentDistX > 10) {
                e.preventDefault();
            }
        }, {passive: false});
        
        element.addEventListener('touchend', function(e) {
            const touchObj = e.changedTouches[0];
            distX = touchObj.pageX - startX;
            distY = touchObj.pageY - startY;
            const elapsedTime = new Date().getTime() - startTime;
            
            // Check if it's a valid swipe left gesture
            if (elapsedTime <= allowedTime && 
                Math.abs(distX) >= threshold && 
                Math.abs(distY) <= restraint && 
                distX < 0) {
                closeMobileMenu();
            }
        }, {passive: true});
        
        // Also enable swipe on overlay
        overlay.addEventListener('touchstart', function(e) {
            const touchObj = e.changedTouches[0];
            startX = touchObj.pageX;
            startTime = new Date().getTime();
        }, {passive: true});
        
        overlay.addEventListener('touchend', function(e) {
            const touchObj = e.changedTouches[0];
            distX = touchObj.pageX - startX;
            const elapsedTime = new Date().getTime() - startTime;
            
            if (elapsedTime <= allowedTime && Math.abs(distX) >= threshold && distX < 0) {
                closeMobileMenu();
            }
        }, {passive: true});
    }
    
    // Focus trap functionality
    function trapFocus(e) {
        if (!sidebar.classList.contains('active')) return;
        
        const focusableElements = sidebar.querySelectorAll(
            'a[href], button, textarea, input[type="text"], input[type="radio"], input[type="checkbox"], select, [tabindex]:not([tabindex="-1"])'
        );
        
        const firstFocusable = focusableElements[0];
        const lastFocusable = focusableElements[focusableElements.length - 1];
        
        if (e.key === 'Tab') {
            if (e.shiftKey) {
                if (document.activeElement === firstFocusable) {
                    lastFocusable.focus();
                    e.preventDefault();
                }
            } else {
                if (document.activeElement === lastFocusable) {
                    firstFocusable.focus();
                    e.preventDefault();
                }
            }
        }
    }
    
    document.addEventListener('keydown', trapFocus);
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
        
        // Hide the original alert
        alert.style.display = 'none';
    });
    
    // if (window.location.pathname === '/') {
    //     setTimeout(() => {
    //         showNotification('Welcome to GradeUP! Notification system is working.', 'info', 3000);
    //     }, 1000);
    // }
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

// Favorites functionality completely removed 

// ===== MOBILE UTILITY FUNCTIONS =====

// Viewport height fix for mobile browsers
function initViewportFix() {
    function setViewportHeight() {
        const vh = window.innerHeight * 0.01;
        document.documentElement.style.setProperty('--vh', `${vh}px`);
    }
    
    setViewportHeight();
    window.addEventListener('resize', throttle(setViewportHeight, 250));
    window.addEventListener('orientationchange', () => {
        setTimeout(setViewportHeight, 500);
    });
}

// Touch device detection and optimization
function initTouchOptimizations() {
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    
    if (isTouchDevice) {
        document.documentElement.classList.add('touch-device');
        
        // Improve touch scrolling on iOS
        document.addEventListener('touchstart', function() {}, { passive: true });
        
        // Prevent zoom on double tap for form inputs
        let lastTouchEnd = 0;
        document.addEventListener('touchend', function(event) {
            const now = (new Date()).getTime();
            if (now - lastTouchEnd <= 300) {
                event.preventDefault();
            }
            lastTouchEnd = now;
        }, false);
        
        // Add touch ripple effect to buttons
        addTouchRipple();
    } else {
        document.documentElement.classList.add('no-touch');
    }
}

// Add ripple effect to buttons on touch
function addTouchRipple() {
    const buttons = document.querySelectorAll('.btn, .nav-link, .university-btn, .user-btn');
    
    buttons.forEach(button => {
        button.addEventListener('touchstart', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.touches[0].clientX - rect.left - size / 2;
            const y = e.touches[0].clientY - rect.top - size / 2;
            
            ripple.style.cssText = `
                position: absolute;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.3);
                transform: scale(0);
                animation: ripple 0.6s linear;
                left: ${x}px;
                top: ${y}px;
                width: ${size}px;
                height: ${size}px;
                pointer-events: none;
            `;
            
            const rippleContainer = this.querySelector('.ripple-container') || this;
            rippleContainer.style.position = 'relative';
            rippleContainer.style.overflow = 'hidden';
            rippleContainer.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        }, { passive: true });
    });
}

// Optimize images for mobile
function initImageOptimization() {
    const images = document.querySelectorAll('img');
    
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    
                    // Lazy load images
                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.removeAttribute('data-src');
                    }
                    
                    // Add loading animation
                    img.classList.add('loaded');
                    observer.unobserve(img);
                }
            });
        });
        
        images.forEach(img => {
            if (img.dataset.src) {
                imageObserver.observe(img);
            }
        });
    }
}

// Handle network status changes
function initNetworkOptimization() {
    if ('connection' in navigator) {
        const connection = navigator.connection;
        
        function updateConnectionStatus() {
            const slowConnection = connection.effectiveType === 'slow-2g' || 
                                 connection.effectiveType === '2g' ||
                                 connection.saveData;
            
            if (slowConnection) {
                document.documentElement.classList.add('slow-connection');
                // Reduce animations and particle effects
                document.querySelectorAll('.particle, .hero-particle, .web-particle').forEach(el => {
                    el.style.display = 'none';
                });
            } else {
                document.documentElement.classList.remove('slow-connection');
            }
        }
        
        updateConnectionStatus();
        connection.addEventListener('change', updateConnectionStatus);
    }
}

// Enhanced form handling for mobile
function initMobileFormEnhancements() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input, textarea, select');
        
        inputs.forEach(input => {
            // Add input type optimizations
            if (input.type === 'email') {
                input.setAttribute('inputmode', 'email');
                input.setAttribute('autocomplete', 'email');
            } else if (input.type === 'tel') {
                input.setAttribute('inputmode', 'tel');
            } else if (input.type === 'search') {
                input.setAttribute('inputmode', 'search');
            }
            
            // Add visual feedback for form validation
            input.addEventListener('blur', function() {
                if (this.validity.valid) {
                    this.classList.remove('invalid');
                    this.classList.add('valid');
                } else if (this.value) {
                    this.classList.remove('valid');
                    this.classList.add('invalid');
                }
            });
            
            // Auto-resize textareas
            if (input.tagName === 'TEXTAREA') {
                input.addEventListener('input', function() {
                    this.style.height = 'auto';
                    this.style.height = this.scrollHeight + 'px';
                });
            }
        });
    });
}

// Performance monitoring for mobile
function initPerformanceMonitoring() {
    if ('performance' in window) {
        // Monitor frame rate
        let frameCount = 0;
        let lastTime = performance.now();
        
        function countFrames() {
            frameCount++;
            const currentTime = performance.now();
            
            if (currentTime - lastTime >= 1000) {
                const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
                
                // If FPS is too low, reduce animations
                if (fps < 30) {
                    document.documentElement.classList.add('low-performance');
                }
                
                frameCount = 0;
                lastTime = currentTime;
            }
            
            requestAnimationFrame(countFrames);
        }
        
        requestAnimationFrame(countFrames);
    }
}

// Initialize all mobile optimizations
function initMobileOptimizations() {
    initViewportFix();
    initTouchOptimizations();
    initImageOptimization();
    initNetworkOptimization();
    initMobileFormEnhancements();
    initPerformanceMonitoring();
}

// Add CSS animations for touch ripple
const rippleCSS = `
@keyframes ripple {
    to {
        transform: scale(4);
        opacity: 0;
    }
}

.touch-device .btn:active,
.touch-device .nav-link:active {
    transform: scale(0.98);
}

.low-performance .particle,
.low-performance .hero-particle,
.low-performance .web-particle,
.slow-connection .particle,
.slow-connection .hero-particle,
.slow-connection .web-particle {
    display: none !important;
}

.img {
    opacity: 0;
    transition: opacity 0.3s ease;
}

.img.loaded {
    opacity: 1;
}

.input.valid {
    border-color: #22c55e;
    box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2);
}

.input.invalid {
    border-color: #ef4444;
    box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2);
}

/* CSS-only viewport height fix */
.hero-section {
    min-height: calc(var(--vh, 1vh) * 100 - var(--top-navbar-height));
}

@media (max-width: 768px) {
    .hero-section {
        min-height: calc(var(--vh, 1vh) * 100 - 60px);
    }
}
`;

// Inject CSS
if (!document.querySelector('#mobile-optimizations-css')) {
    const style = document.createElement('style');
    style.id = 'mobile-optimizations-css';
    style.textContent = rippleCSS;
    document.head.appendChild(style);
}

// Initialize mobile optimizations when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMobileOptimizations);
} else {
    initMobileOptimizations();
} 