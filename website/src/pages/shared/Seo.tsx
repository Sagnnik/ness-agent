import { useEffect } from 'react'

export const SITE_URL = 'https://nessagent.dev'
export const SITE_NAME = 'Ness Agent'

export interface SeoProps {
  title: string
  description: string
  path: string
  type?: 'website' | 'article'
  image?: string
  publishedTime?: string
}

function absoluteUrl(pathOrUrl: string, trailingSlash = false) {
  try {
    const url = new URL(pathOrUrl, SITE_URL)
    if (trailingSlash && pathOrUrl.startsWith('/') && url.pathname !== '/') {
      url.pathname = `${url.pathname.replace(/\/+$/, '')}/`
    }
    return url.toString()
  } catch {
    return pathOrUrl
  }
}

function upsertMeta(attribute: 'name' | 'property', key: string, content: string) {
  let element = document.head.querySelector<HTMLMetaElement>(
    `meta[${attribute}="${key}"]`,
  )
  if (!element) {
    element = document.createElement('meta')
    element.setAttribute(attribute, key)
    document.head.append(element)
  }
  element.content = content
}

function removeMeta(attribute: 'name' | 'property', key: string) {
  document.head.querySelector(`meta[${attribute}="${key}"]`)?.remove()
}

function upsertCanonical(href: string) {
  let element = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]')
  if (!element) {
    element = document.createElement('link')
    element.rel = 'canonical'
    document.head.append(element)
  }
  element.href = href
}

export function Seo({
  title,
  description,
  path,
  type = 'website',
  image,
  publishedTime,
}: SeoProps) {
  useEffect(() => {
    const url = absoluteUrl(path, true)
    const imageUrl = image ? absoluteUrl(image) : ''

    document.title = title
    upsertCanonical(url)
    upsertMeta('name', 'description', description)
    upsertMeta('name', 'robots', 'index,follow,max-image-preview:large')
    upsertMeta('property', 'og:site_name', SITE_NAME)
    upsertMeta('property', 'og:type', type)
    upsertMeta('property', 'og:title', title)
    upsertMeta('property', 'og:description', description)
    upsertMeta('property', 'og:url', url)
    upsertMeta('name', 'twitter:card', imageUrl ? 'summary_large_image' : 'summary')
    upsertMeta('name', 'twitter:title', title)
    upsertMeta('name', 'twitter:description', description)

    if (imageUrl) {
      upsertMeta('property', 'og:image', imageUrl)
      upsertMeta('name', 'twitter:image', imageUrl)
    } else {
      removeMeta('property', 'og:image')
      removeMeta('name', 'twitter:image')
    }

    if (type === 'article' && publishedTime) {
      upsertMeta('property', 'article:published_time', publishedTime)
      upsertMeta('property', 'article:author', 'Sagnnik Biswas')
    } else {
      removeMeta('property', 'article:published_time')
      removeMeta('property', 'article:author')
    }

    let structuredData = document.head.querySelector<HTMLScriptElement>(
      '#ness-seo-jsonld',
    )
    if (!structuredData) {
      structuredData = document.createElement('script')
      structuredData.id = 'ness-seo-jsonld'
      structuredData.type = 'application/ld+json'
      document.head.append(structuredData)
    }

    const schema =
      type === 'article'
        ? {
            '@context': 'https://schema.org',
            '@type': 'Article',
            headline: title,
            description,
            datePublished: publishedTime,
            author: { '@type': 'Person', name: 'Sagnnik Biswas' },
            mainEntityOfPage: url,
            ...(imageUrl ? { image: [imageUrl] } : {}),
          }
        : {
            '@context': 'https://schema.org',
            '@type': 'WebSite',
            name: SITE_NAME,
            description,
            url: SITE_URL,
          }
    structuredData.textContent = JSON.stringify(schema)

    return () => {
      structuredData?.remove()
    }
  }, [description, image, path, publishedTime, title, type])

  return null
}
