from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # Headless=False -> browser dikhega
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    
    # Tumhare backend ka URL
    page.goto("http://localhost:8080/")
    
    print("Browser opened. Check the page!")

    # Browser open rakho tab tak jab tak Enter press na karo
    input("Press Enter to close the browser...")
    browser.close()
